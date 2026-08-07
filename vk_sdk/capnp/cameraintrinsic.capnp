@0xe83fc74cfb51c4d6;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("vkc");

struct PinholeModel {
    fx @0 :Float32;
    fy @1 :Float32;
    cx @2 :Float32;
    cy @3 :Float32;
}

struct DoubleSphereModel {
    pinhole @0 :PinholeModel;
    xi @1 :Float32;
    alpha @2 :Float32;
}

struct KB4Model {
    pinhole @0 :PinholeModel;
    k1 @1 :Float32;
    k2 @2 :Float32;
    k3 @3 :Float32;
    k4 @4 :Float32;
}

struct RadtanModel {
    pinhole @0 :PinholeModel;
    k1 @1 :Float32;
    k2 @2 :Float32;
    p1 @3 :Float32;
    p2 @4 :Float32;
    k3 @5 :Float32;
    k4 @6 :Float32;
    k5 @7 :Float32;
    k6 @8 :Float32;
}

struct CameraIntrinsic {
    
    pinhole @0 :PinholeModel;
    ds @1 :DoubleSphereModel;
    kb4 @2 :KB4Model;
    radtan8 @5 :RadtanModel; # added 2024-09

    # metadata
    rectified @3 :Bool;
    lastModified @4 :UInt64; # UTC time in nanosecond
}