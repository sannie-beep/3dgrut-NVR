@0x9b108344e15f6346;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("vkc");

struct Point2Df {
    x @0 :Float32;
    y @1 :Float32;
    score @2 :Float32; # optional score
}

struct ExternalPointDetection {
    header @0 :import "header.capnp".Header;
    
    streamName @1 :Text; # camera stream name
    points @2 :List(Point2Df);  # normalized image coordinates of detected points
}