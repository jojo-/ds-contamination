#!/usr/bin/python3

################################################################################
# REMONDIS
#
# Automated license plate recognition and traffic count with DeepStream.
#
# Version: 04 May 2022
# Author : Johan Barthelemy - johan@uow.edu.au
#
# License: MIT
# Copyright (c) 2022 Johan Barthelemy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################ 

################################################################################
# This work relies on some functions initially written by NVIDIA and modified
# by the author of REMONDIS ALPR:
# - cb_newpad
# - decodebin_child_added
# - create_source_bin
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys
import configparser
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
from datetime import datetime
import pyds
import numpy as np
import cv2
import json
import hashlib
import os
import uuid
import threading
import netifaces
import ssl

# Dictionnary to store the data
dict_detection = dict()

# Primary inference engine
PGIE_CLASS_ID_CONTAMINATION = 0

# Display output
DISPLAY_OUT = 1

# FPS
FPS_OUT = 0
fps_stream = GETFPS(0)

# Config DATA
DEVICE_NAME = netifaces.ifaddresses(netifaces.interfaces()[1])[netifaces.AF_LINK][0]['addr']
DATA_ENABLE = 0

# Sending the collected
def send_data_thread():
   
    # Umair to implement the data transmission logic

    return True                       

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not audio
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick an NVIDIA decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   
    if(is_aarch64() and name.find("nvv4l2decoder") != -1):
        print("Seting bufapi_version\n")
        Object.set_property("bufapi-version",True)
        
def create_source_bin(uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin"
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin        

def osd_sink_pad_buffer_probe(pad,info,u_data):
    
    frame_number = 0
    #Intiallizing object counter with 0    
    obj_counter = {
        PGIE_CLASS_ID_CONTAMINATION: 0,          
    }    

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))    

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
           
        except StopIteration:
            break
            
        frame_number = frame_meta.frame_num        
        l_obj        = frame_meta.obj_meta_list

        first_obj = True
        now_ts = datetime.now()

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)                                
            except StopIteration:
                break
                                                                                                                                         
            obj_counter[obj_meta.class_id] += 1
            
            # Current object is a vehicle
            if (obj_meta.class_id == PGIE_CLASS_ID_CONTAMINATION):                
                                                                                           
                if obj_meta.object_id not in dict_detection:
                    dict_detection[obj_meta.object_id] = (now_ts, now_ts)
                else:
                    ts_in = dict_detection[obj_meta.object_id][0]
                    dict_detection[obj_meta.object_id] = (ts_in, now_ts)                                                        
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # saving image if a contamination has been detected
        if obj_counter[PGIE_CLASS_ID_CONTAMINATION] > 0:
            n_frame   = pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
            frame_image = np.array(n_frame,copy=True,order='C')            
            cv2.imwrite("out\img-" + DEVICE_NAME + "-" +  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg", frame_image)
                   
        if DATA_ENABLE:
            if frame_number % DATA_RATE == 0:
                t = threading.Thread(target=send_data_thread)
                t.start()
               
        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} detection_count={}".format(frame_number, obj_counter[PGIE_CLASS_ID_CONTAMINATION])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        #print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        if FPS_OUT == 1:
            fps_stream.get_fps()

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	

def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <config file>\n" % args[0])
        sys.exit(1)

    # Reading configuration file
    config_app = configparser.ConfigParser()
    config_app.read(args[1])
    config_app.sections()
    
    for key in config_app['device']:
        if key == 'name':
            global DEVICE_NAME
            DEVICE_NAME = config_app.get('device', key)

    for key in config_app['source']:
        if key == 'mode':
            MODE_INPUT = config_app.getint('source', key)
        if key == 'uri':
            URI_INPUT = config_app.get('source', key)

    for key in config_app['output']:
        if key == 'enable-display':
            global DISPLAY_OUT
            DISPLAY_OUT = config_app.getint('output', key)
        if key == 'enable-rtsp':
            RTSP_OUT = config_app.getint('output', key)
        if key == 'enable-file-out':
            FILE_OUT = config_app.getint('output', key)       
        if key == 'enable-fps':
            global FPS_OUT
            FPS_OUT = config_app.getint('output', key)

    for key in config_app['data']:
        if key == 'data-enable':
            global DATA_ENABLE
            DATA_ENABLE = config_app.getint('data', key)        
        if key == 'data-rate':
            global DATA_RATE
            DATA_RATE = config_app.getint('data', key)        
            
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create GStreamer Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    source = None
    # ... for webcam
    caps_v4l2src    = None
    vidconvsrc      = None
    nvvidconvsrc    = None
    caps_vidconvsrc = None    

    if MODE_INPUT == 0:

        # Source element for reading from a webcam
        print("Creating Source \n ")
        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")

        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        if not caps_v4l2src:
            sys.stderr.write(" Unable to create v4l2src capsfilter \n")
        
        # videoconvert to make sure a superset of raw formats are supported
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        if not vidconvsrc:
            sys.stderr.write(" Unable to create videoconvert \n")

        # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
        nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        if not nvvidconvsrc:
            sys.stderr.write(" Unable to create Nvvideoconvert \n")

        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        if not caps_vidconvsrc:
            sys.stderr.write(" Unable to create capsfilter \n")

    elif MODE_INPUT == 1:
        print("Creating Source \n ")
        source = create_source_bin(URI_INPUT)
        if not source:
            sys.stderr.write("Unable to create source bin \n")       
        
    else:
        sys.stderr.write("INPUT_MODE not suppported: {}".format(MODE_INPUT))
        sys.exit(1)

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    
    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Add a tracker
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
   
    # Sink
    print("Creating sink(s) \n")

    # For splitting the pipeline if sinks in parallel
    tee=None
         
    # No sinks, fake output       
    if DISPLAY_OUT + RTSP_OUT + FILE_OUT == 0:
        print(" fakesink selected")
        sink_fake = Gst.ElementFactory.make("fakesink", "fake-sink")    
        if not sink_fake:
            sys.stderr.write(" Unable to create fake display sink \n")
        sink_fake.set_property('sync', False)
    
    else:                    
        tee=Gst.ElementFactory.make("tee", "nvsink-tee")
        if not tee:
            sys.stderr.write(" Unable to create tee \n")
            
    if DISPLAY_OUT == 1:
        print(" display sink selected")
        
        queue_disp=Gst.ElementFactory.make("queue", "nvtee-q-disp")
        if not queue_disp:
            sys.stderr.write(" Unable to create queue for display \n")
            
        # Add a transformation before rendering rendering the osd output if Jetson platform
        if is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        
        sink_disp = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")    
        if not sink_disp:
            sys.stderr.write(" Unable to create display sink \n")    
        # Set sync = false to avoid late frame drops at the display-sink
        sink_disp.set_property('sync', False)        
               
    if FILE_OUT == 1:
    
        print("Creating file sink")
              
        queue_file=Gst.ElementFactory.make("queue", "nvtee-q-file")
        if not queue_file:
            sys.stderr.write(" Unable to create queue for file \n")
        
        nvvidconv_postosd_f = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd_f")
        if not nvvidconv_postosd_f:
            sys.stderr.write(" Unable to create nvvidconv_postosd_f \n")
                        
        encoder_f = Gst.ElementFactory.make("nvv4l2h265enc", "encoder_f")        
        if not encoder_f:
            sys.stderr.write(" Unable to create H265 encoder for file")
            
        if is_aarch64():
            encoder_f.set_property('preset-level', 1)
            encoder_f.set_property('insert-sps-pps', 1)
            encoder_f.set_property('bufapi-version', 1)
            
        parser_f = Gst.ElementFactory.make("h265parse", "h265parser_f")        
        if not parser_f:
            sys.stderr.write(" Unable to create H265 parser for file")
            
        qtmux = Gst.ElementFactory.make("qtmux", "muxer")
        if not encoder_f:
            sys.stderr.write(" Unable to create muxer")   
            
        sink_file = Gst.ElementFactory.make("filesink", "filesink")
        sink_file.set_property('location', 'out/out.h265')

    if RTSP_OUT == 1:
    
        print("Creating RTSP sink")
              
        queue_rtsp=Gst.ElementFactory.make("queue", "nvtee-q-rtsp")
        if not queue_rtsp:
            sys.stderr.write(" Unable to create queue for rtsp \n")
        
        nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
        if not nvvidconv_postosd:
            sys.stderr.write(" Unable to create nvvidconv_postosd \n")
        
        caps_rtsp = Gst.ElementFactory.make("capsfilter", "filter")
        if not caps_rtsp:
            sys.stderr.write(" Unable to create caps filter\n")            
        caps_rtsp.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
                
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")        
        if not encoder:
            sys.stderr.write(" Unable to create H265 encoder")
            
        if is_aarch64():
            encoder.set_property('preset-level', 1)
            encoder.set_property('insert-sps-pps', 1)
            encoder.set_property('bufapi-version', 1)
                    
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")    
        if not rtppay:
            sys.stderr.write(" Unable to create rtppay")
        
        updsink_port_num = 5400
        sink_udp = Gst.ElementFactory.make("udpsink", "udpsink")
        if not sink_udp:
            sys.stderr.write(" Unable to create udpsink")
    
        sink_udp.set_property('host', '224.224.255.255')
        sink_udp.set_property('port', updsink_port_num)
        sink_udp.set_property('async', False)
        sink_udp.set_property('sync', True)
                                                
    # Set properties of source
    if MODE_INPUT == 0:
        print("Playing cam {} ".format(URI_INPUT))
        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        source.set_property('device', URI_INPUT)                       
    
    # Set properties of streamux
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)    
    if URI_INPUT.find("rtsp://") == 0 :
        print("Source is live")
        streammux.set_property('live-source', 1)
    
    # Use CUDA unified memory in the pipeline so frames can be easily accessed on CPU
    if not is_aarch64():        
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)          

    # Set properties of inference engines
    pgie.set_property('config-file-path', "config/config_primary_inference_engine.txt")
   
        
    # Set properties of tracker
    config_tracker = configparser.ConfigParser()
    config_tracker.read('config/config_tracker.txt')
    config_tracker.sections()
    
    for key in config_tracker['tracker']:
        if key == 'tracker-width' :
            tracker_width = config_tracker.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config_tracker.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config_tracker.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config_tracker.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config_tracker.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config_tracker.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        
    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    
    if MODE_INPUT == 0:
        pipeline.add(caps_v4l2src)
        pipeline.add(vidconvsrc)
        pipeline.add(nvvidconvsrc)
        pipeline.add(caps_vidconvsrc)
   
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
      
    if DISPLAY_OUT + RTSP_OUT + FILE_OUT == 0:
        pipeline.add(sink_fake)    
    else:
        pipeline.add(tee)
        if DISPLAY_OUT == 1:
            pipeline.add(queue_disp)
            if is_aarch64():
                pipeline.add(transform)
            pipeline.add(sink_disp)          
        if RTSP_OUT == 1:    
            pipeline.add(queue_rtsp)
            pipeline.add(nvvidconv_postosd)
            pipeline.add(caps_rtsp)
            pipeline.add(encoder)
            pipeline.add(rtppay)
            pipeline.add(sink_udp)
        if FILE_OUT == 1:
            pipeline.add(queue_file)
            pipeline.add(nvvidconv_postosd_f)
            pipeline.add(encoder_f)
            pipeline.add(parser_f)
            pipeline.add(qtmux)
            pipeline.add(sink_file)         

    # Linking the elements together
    # source -> mux -> nvinfer (pri) -> tracker -> nvinfer (sec) -> nvinfer (sec) -> nvvideoconvert -> nvosd -> sink
    print("Linking elements in the Pipeline \n")

    if MODE_INPUT == 0:
        source.link(caps_v4l2src)
        caps_v4l2src.link(vidconvsrc)
        vidconvsrc.link(nvvidconvsrc)
        nvvidconvsrc.link(caps_vidconvsrc)
    
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = None
    if MODE_INPUT == 0:
        srcpad = caps_vidconvsrc.get_static_pad("src")
    elif MODE_INPUT == 1:
        srcpad = source.get_static_pad("src")   
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)

    streammux.link(pgie)   
    pgie.link(tracker)
    tracker.link(nvvidconv)    
    nvvidconv.link(nvosd)
    
    if RTSP_OUT + DISPLAY_OUT + FILE_OUT == 0:
        nvosd.link(sink_fake)
    
    else:
        nvosd.link(tee)
        
        if DISPLAY_OUT == 1:
            sink_pad_q_disp = queue_disp.get_static_pad("sink")
            tee_disp_pad = tee.get_request_pad('src_%u')
            tee_disp_pad.link(sink_pad_q_disp)
            
            if not tee_disp_pad:
                sys.stderr.write("Unable to get requested tee src pads for display\n")
       
            if is_aarch64():
                queue_disp.link(transform)
                transform.link(sink_disp)
            else:
                queue_disp.link(sink_disp)
                
        if RTSP_OUT == 1:
            sink_pad_q_rtsp  = queue_rtsp.get_static_pad("sink")
            tee_rtsp_pad = tee.get_request_pad('src_%u')
            tee_rtsp_pad.link(sink_pad_q_rtsp)
            
            if not tee_rtsp_pad:
                sys.stderr.write("Unable to get requested tee src pads for rtsp\n")
            
            queue_rtsp.link(nvvidconv_postosd)
            nvvidconv_postosd.link(caps_rtsp)
            caps_rtsp.link(encoder)
            encoder.link(rtppay)
            rtppay.link(sink_udp)                                               
        
        if FILE_OUT == 1:
            sink_pad_q_file  = queue_file.get_static_pad("sink")
            tee_file_pad = tee.get_request_pad('src_%u')
            tee_file_pad.link(sink_pad_q_file)
            
            if not tee_file_pad:
                sys.stderr.write("Unable to get requested tee src pads for file\n")
            
            queue_file.link(nvvidconv_postosd_f)
            nvvidconv_postosd_f.link(encoder_f)
            encoder_f.link(parser_f)
            parser_f.link(qtmux)
            qtmux.link(sink_file)       
        

    # Create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
        
    # Start streaming if RTSP is enabled
    if RTSP_OUT == 1:
        rtsp_port_num = 8554        
        server = GstRtspServer.RTSPServer.new()
        server.props.service = "%d" % rtsp_port_num
        server.attach(None)
        
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H265, payload=96 \" )" % (updsink_port_num))
        factory.set_shared(True)
        server.get_mount_points().add_factory("/live", factory)
        
        print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/live ***\n\n" % rtsp_port_num)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
        
    # Transmitting the final data
    if DATA_ENABLE:
        print("INFO: final data transmission")      
        send_data_thread()
                   
    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
