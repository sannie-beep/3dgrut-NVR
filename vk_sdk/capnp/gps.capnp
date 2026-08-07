@0xec40a6b023efec28;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("vkc");

struct GPS {
    header @0 :import "header.capnp".Header;

    lat @1 :Float64;
    long @2 :Float64;
    alt @3 :Float64;

    speed2d @4 :Float64;
    speed3d @5 :Float64;

    dop @6 :Float32;  
    # Disregard point if dop (Dilution of precision) value above 20
}