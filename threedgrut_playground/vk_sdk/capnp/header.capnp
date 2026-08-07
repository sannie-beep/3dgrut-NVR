@0xc61974417b74b4cc;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("vkc");

struct HeaderMetadata {
    key @0 :Text;
    value @1 :Float64;
}

struct Header {

    seq @0 :UInt64;
    stampMonotonic @1 :UInt64;      # Monotonic time (nsec) when data is created (captured, measured etc); host clock domain
    frameId @2 :Text;               # Used by camera driver for identification of the unit serial number
    stampMonotonicDevice @3 :UInt64;    # corresponding time (nsec) to stampMonotonic, in ISP device time
    latencyDevice @4 :UInt64;       # Latency introduced from device sensor capture until userspace reception at host CPU.
    clockOffset @5 : Int64;         # Offset to be added to stampMonotonic to obtain system time in nsec (wall-clock time).

    metadata @6 :List(HeaderMetadata); # a list of metadata associated
}