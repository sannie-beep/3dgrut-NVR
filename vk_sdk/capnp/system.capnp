@0xdabfd4b1be2a895b;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("vkc");

#using Rust = import "/capnp/rust.capnp";
#$Rust.parentModule("capnp");

#########################################################################
# Commands
#########################################################################

struct ManagerCommand {
    marker @0: UInt64;                          # User-specified number that will be attached to corresponding returning `Message`s.
                                                # The number `0` should not be used as it is reserved to signal that a message does not  
                                                # correspond to any previously issued command by this client.
    variant: union {
        system @1: SystemCommand;               # Command relating to device system.
        network @2: NetworkCommand;             # Command relating to device network configuration.
        time @3: TimeCommand;                   # Command relating to time synchronization.
        cameraDriver @4: CameraDriverCommand;   # Command relating to running camera driver processes.
        vio @5: VioCommand;                     # Command relating to running VIO processes.
        recording @6: RecordingCommand;         # Command relating to recording.
        script @7: ScriptCommand;               # Command relating to running scripts.
        viewer @8: ViewerCommand;               # Command relating to running viewer.
        playback @9: PlaybackCommand;           # Command relating to running playback.
        calibration @10: CalibrationCommand;    # Command relating to flashing calibration configs.
        reprojection @11: ReprojectionCommand;  # Command relating to running reprojection tool.
    }
}

struct SystemCommand {
    variant: union {
        update @0: SystemUpdate;                    # Command: Update the system.
        uploadUpdateBegin @1 :UploadUpdateBegin;    # Command: Record target path and expected size
        uploadUpdateChunk @2 :UploadUpdateChunk;    # Command: Accepts chunks
        uploadUpdateFinish @3 :UploadUpdateFinish;  # Command: Verifies full size
    }
}

struct NetworkCommand {
    variant: union {                                # Variant of command issued.
        fetchSettings @0: Void;                     # Command: Fetch network settings of the system.
        optConfiguration @1: NetworkConfiguration;  # Command: Persistently set the network configuration of the system.
    }
}

struct TimeCommand {
    variant: union {                    # Variant of command issued.
        stop @0: Void;                  # Command: Stop the time synchronization process.
        reload @1: Void;                # Command: Reload the time synchronization process.
        fetchSettings @2: Void;         # Command: Fetch all settings of the time synchronization process.
        optAutostart @3: Bool;          # Command: Persistently set the time synchronization process to auto start on boot.
        optRole @4: TimeRole;           # Command: Persistently set the role of the system in the time synchronization process.
    }
}

struct CameraDriverCommand {
    target @0: UInt32;                  # Index of the target camera driver this command applies to.
    variant: union {                    # Variant of command issued.
        stop @1: Void;                  # Command: Stop the camera driver process.
        reload @2: List(UInt8);         # Command: Reload the camera driver process.
        fetchSettings @3: Void;         # Command: Fetch all settings of the camera driver.
        optAutostart @4: Bool;          # Command: Persistently set the camera driver to auto start on boot.
        optWaitForTimeSync @5: Bool;    # Command: Persistently set the camera driver to wait for time synchronization before running.
        optStartConfiguration @6: Text; # Command: Persistently set the configuration to be used when a camera driver is started/reloaded.
        deleteLogFile @7: Text;         # Command: Delete the specified log file.
    }
}

struct VioCommand {
    target @0: UInt32;                  # Index of the target vio this command applies to.
    variant: union {                    # Variant of command issued.
        stop @1: Void;                  # Command: Stop the VIO process.
        reload @2: List(UInt8);                # Command: Reload the VIO process using the configuration name.
        fetchSettings @3: Void;         # Command: Fetch all settings of the VIO.
        optAutostart @4: Bool;          # Command: Persistently set the VIO to auto start on boot.
        optStartConfiguration @5: Text; # Command: Persistently set the configuration to be used when a VIO is started/reloaded.
        deleteLogFile @6: Text;         # Command: Delete the specified log file.
    }
}

struct RecordingCommand {
    variant: union {                                        # Variant of command issued.
        stop @0: Void;                                      # Command: Stop the recording process.
        start @1: List(UInt8);                           # Command: Start the recording process.
        fetchSettings @2: Void;                             # Command: Fetch all settings of the recording process.
        optSelectedTopicSet @3: Text;                       # Command: Persistently set the selected topic set for recording.
        optCompressionAlgorithm @4: CompressionAlgorithm;   # Command: Persistently set the compression algorithm used when storing a recording.
        optCompressionLevel @5: CompressionLevel;           # Command: Persistently set the compression level used when storing a recording.
        optRecordManager @8: Bool;                          # Command: Persistently set whether manager messages are recorded as well.
        fetchRecordingInfo @6: Text;                        # Command: Fetch metadata information of the specified recording file.
        deleteRecording @7: Text;                           # Command: Delete the specified recording file.
        deleteLogFile @9: Text;                              # Command: Delete the specified log file.
    }
}

struct PlaybackCommand {
    variant: union {                                        # Variant of command issued.
        stop @0: Void;                                      # Command: Stop the recording process.
        start @1: Void;                                     # Command: Start the recording process.
        fetchSettings @2: Void;                             # Command: Fetch all settings of the recording process.
        optSelectedRecordingFile @3: Text;                  # Command: Persistently set the selected recording file for playback.
        deleteLogFile @4: Text;                             # Command: Delete the specified log file.
    }
}

struct ScriptCommand {
    variant: union {
        fetchAvailableScripts @0: Void; # Command: Fetch all available scripts.
        execute @1: ExecuteScript;      # Command: Execute a script.
        kill @2: UInt32;                # Command: Kill a running script process via their id.
    }
}

struct ViewerCommand {
    variant: union {                                        # Variant of command issued.
        stop @0: Void;                                      # Command: Stop the recording process.
        start @1: Void;                                     # Command: Start the recording process.
        fetchSettings @2: Void;                             # Command: Fetch all settings of the recording process.
        optSelectedTopicSet @3: Text;                       # Command: Persistently set the selected topic set for recording.
        optViewerDataMode @4: ViewerDataMode;               # Command: Persistently set the Data mode for vk_viewer.
        optWebViewer @5: Bool;                              # Command: Persistently set web viewer option
        optFramerateFactor @6: UInt32;                      # Command: Persistently set framerate factor for vk_viewer.
    }
}

struct CalibrationCommand {
    variant: union {
        flashCalibration @0: FlashCalibration;              #Execute calibration command
        fetchAvailableConfigs @1: FetchAvailableConfigs;    # fetchAvailableScripts
        deployJsonConfig @2: DeployJsonConfig;              #Copyjsontodevice
        fetchFromServer @3 : FetchFromServer;               #Fetch json from server
        login @4 : LoginArgs;                               #login
        dumpCalibrationConfig @5 : DumpCalibrationConfig;               #Dump
    }
}

struct ReprojectionCommand {
    variant: union {
        start @0: Void;                         # Command: Start vk_reproj_runner in local mode.
        stop @1: Void;                          # Command: Stop vk_reproj_runner.
        fetchSettings @2: Void;                # Command: Fetch reprojection settings and available configs.
        optSelectedConfiguration @3: Text;     # Command: Persistently set the reprojection config used on start.
        optUnitSerial @4: Text;                # Command: Persistently set the unit serial passed to the runner.
        fetchAvailableRuns @5: Void;           # Command: Fetch saved reprojection run directories for the active output path.
        fetchRunResults @6: Text;              # Command: Fetch contents of the saved run result YAML files.
        deleteRun @7: Text;                    # Command: Delete the specified saved reprojection run directory.
    }
}

struct LoginArgs {
  username @0 : Text;
  password @1 : Text;
}

struct FetchFromServer {
    productId @0 : Text;
}

struct DeployJsonConfig {
    fileName @0: Text;       # name
    directory @1: Text;      # dir to save
    content @2: Data;        # jsonfile
}

struct FetchAvailableConfigs {
    directory @0 :Text;
}

struct FlashCalibration {
    cameraType @0: Text;   # "cv52" or "luxonis"
    configFile @1: Text;   # path of config file
}

struct DumpCalibrationConfig {
    cameraType @0 : Text;   # "cv52" or "luxonis"
}


#########################################################################
# Messages
#########################################################################

struct ManagerMessage {
    marker @0: UInt64;                          # User-specified marker that corresponds this message to a previously issued command.
                                                # If this value is `0`, this message does not correspond to any previously issued command.
    variant: union {
        commandOk @1: Void;                     # Signals ok to previously issued command.
        commandError @2: Text;                  # Signals error to previously issued command.
        system @3: SystemMessage;               # Message relating to device system.
        network @4: NetworkMessage;             # Message relating to device network configuration.
        time @5: TimeMessage;                   # Message relating to time synchronization.
        cameraDriver @6: CameraDriverMessage;   # Message relating to camera driver processes.
        vio @7: VioMessage;                     # Message relating to VIO processes.
        recording @8: RecordingMessage;         # Message relating to the recording process.
        script @9: ScriptMessage;               # Message relating to running scripts.
        log @10: Text;                          # Log message from the manager itself.
        viewer @11: ViewerMessage;              # Message relating to viewer process.
        playback @12: PlaybackMessage;          # Message relating to the playback process.
        calibration @13: CalibrationMessage;    # Message relating to the calibration process.
        reprojection @14: ReprojectionMessage;  # Message relating to the reprojection process.
    }
}

struct SystemMessage {
    variant: union {
        info @0: SystemInfo;        # [Periodic] Report: Current system information.
        unused @1: Void;            # Dummy field. Do not use.
    }
}

struct NetworkMessage {
    variant: union {
        enabled @0: Bool;                       # Report: Informs whether network can be modified through commands.
        configuration @1: NetworkConfiguration; # Report: Change in the network configuration.
    }
}

struct CalibrationMessage {
    variant: union {
        logMessage @0: Text;                          # Realtime output
        availableConfigFiles @1: List(Text);          # json files at /home/$USER 
    }
}

struct TimeMessage {
    variant: union {
        status @0: TimeStatus;          # [Periodic] Report: Current status of the time synchronization process.
        inconsistent @4: Void;          # Report: Possible inconsistency in clock, can happen if there is interference between devices.
                                        #         Note: Reported only if time status is `runningSynchronized`.
        logMessage @1: Text;            # Report: New line of log message produced by the time synchronization process.
        optAutostart @2: Bool;          # Report: Change in the persistent autostart setting of the time synchronization process.
        optRole @3: TimeRole;           # Report: Change in the persistent role setting of the system in the time synchronization process.
    }
}

struct CameraDriverMessage {
    target @0: UInt32;                          # Index of the target camera driver this message refers to.
    variant: union {
        heartbeat @1: CameraDriverHeartbeat;    # [Periodic] Heartbeat from the camera driver.
        logMessage @2: Text;                    # Report: New line of warning or error produced by the camera driver process.
        availableConfigurations @3: List(Text); # Report: Available configurations that can be used when starting the camera driver.
        availableLogFiles @4: List(Text);       # Report: Available log files that can be fetched via HTTP.
        optAutostart @5: Bool;                  # Report: Change in the persistent autostart setting of the camera driver.
        optWaitForTimeSync @6: Bool;            # Report: Change in the persistent wait-time-synchronization setting of the camera driver.
        optStartConfiguration @7: Text;         # Report: Change in the persistent start configuration of the camera driver.
        optCpuConfiguration @8: UInt16;         # Report: Change in the persistent cpu configuration of the camera driver.
    }
}

struct VioMessage {
    target @0: UInt32;                          # Index of the target VIO this message refers to.
    variant: union {
        heartbeat @1: VioHeartbeat;             # [Periodic] Heartbeat from the VIO process.
        logMessage @2: Text;                    # Report: New line of warning or error produced by the VIO process.
        availableConfigurations @3: List(Text); # Report: Available configurations that can be used when starting the VIO process.
        availableLogFiles @4: List(Text);       # Report: Available log files that can be fetched via HTTP.
        optAutostart @5: Bool;                  # Report: Change in the persistent autostart setting of the VIO.
        optStartConfiguration @6: Text;         # Report: Change in the persistent start configuration of the VIO.
        optCpuConfiguration @7: UInt16;         # Report: Change in the persistent cpu configuration of the VIO.
    }
}

struct RecordingMessage {
    variant: union {
        status @0: RecordStatus;                                # [Periodic] Report: Current status of the recording process.
        logMessage @1: Text;                                    # Report: New line of log message prduced by the recording process.
        availableTopicSets @2: List(Text);                      # Report: Available topic sets that can be specified for recorded.
        availableRecordings @3: List(Text);                     # Report: Available log files that can be fetched via HTTP.
        optSelectedTopicSet @4: Text;                           # Report: Change in persistent selected topic set setting.
        optCompressionAlgorithm @5: CompressionAlgorithm;       # Report: Change in persistent compression algorithm setting.
        optCompressionLevel @6: CompressionLevel;               # Report: Change in persistent compression level setting.
        optRecordManager @8: Bool;                              # Report: Change in persistent setting of whether manager messages are recorded.
        recordingInfo @7: Text;                                 # Report: Recording info of the previously requested recording.
        availableLogFiles @9: List(Text);                       # Report: Available log files that can be fetched via HTTP.
        optcpucoreconfiguration @10:UInt16;                     # Report: Change in the persistent cpu configuration of the record .
    }
}

struct PlaybackMessage {
    variant: union {
        status @0: PlaybackStatus;                              # [Periodic] Report: Current status of the playback process.
        logMessage @1: Text;                                    # Report: New line of log message prduced by the playback process.
        availableRecordings @2: List(Text);                     # Report: Available recording files.
        optSelectedRecordingFile @3: Text;                      # Report: Recording file to playback.
        availableLogFiles @4: List(Text);                       # Report: Available log files that can be fetched via HTTP.
    }
}

struct ScriptMessage {
    variant: union {
        heartbeat @0: ScriptHeartbeat;          # [Periodic] Report: Status of a executed script.
        availableScripts @1: List(Text);        # Report: Available scripts to be executed.
        stdout @2: ScriptLog;                   # Report: New line of log from stdout of the script process. 
        stderr @3: ScriptLog;                   # Report: New line of log from stderr of the script process. 
    }
}

struct ViewerMessage {
    variant: union {
        status @0: ViewerStatus;                                # [Periodic] Report: Current status of the recording process.
        logMessage @1: Text;                                    # Report: New line of log message prduced by the recording process.
        availableTopicSets @2: List(Text);                      # Report: Available topic sets that can be specified for recorded.
        optSelectedTopicSet @3: Text;                           # Report: Change in persistent selected topic set setting.
        optViewerDataMode @4: ViewerDataMode;                   # Report: Change in persistent viewer Data mode.
        optWebViewer @5: Bool;                                  # Report: Change in persistent setting of web viewer mode.
        optFramerateFactor @6: UInt32;                          # Report: Change in persistent framerate factor.
    }
}

struct ReprojectionMessage {
    variant: union {
        status @0: ReprojectionStatus;               # [Periodic] Report: Current status of the reprojection runner.
        logMessage @1: Text;                         # Report: New line of log message produced by the reprojection runner.
        availableConfigurations @2: List(Text);     # Report: Available reprojection config files.
        optSelectedConfiguration @3: Text;          # Report: Change in persistent selected reprojection config.
        optUnitSerial @4: Text;                     # Report: Change in persistent unit serial.
        installedVersion @5: Text;                  # Report: Installed reprojection package version, if any.
        availableRuns @6: List(Text);               # Report: Saved reprojection run directories for the active output path.
        depthRunResults @7: Text;                   # Report: Full contents of *_DEPTH_*.yaml files for a saved run.
        reprojRunResults @8: Text;                  # Report: Full contents of *_REPROJ_*.yaml files for a saved run.
    }
}

#########################################################################
# Miscellaneous
#########################################################################

struct SystemInfo {
    cpu @0: List(CpuInfo);              # CPU information.
    ram @1: RamInfo;                    # RAM information.
    disk @2: DiskInfo;                  # Disk information.
    temperature @3: TemperatureInfo;    # Temperature information.
    datetime @4: Text;                  # Date/time string in "YYYY-MM-DD hh:mm:ss" format.
    managerVersion @5: Text;            # Version of DP Manager.
    modulesVersion @6: Text;            # Version of DP Modules.
}

struct SystemUpdate {
    packages @0: List(Text);            # Packages or components to update.
    username @1: Text;                  # Customer account username to the update server.
    password @2: Text;                  # Customer account password to the update server.
    preserveSettings @3: Bool;          # Keep the previous settings.yaml values, but add new keys if available
    useLocalFile @4 :Bool;              # true = don't login/download, use local .deb
    localFilePath @5 :Text;             # optional path on the DEVICE to the .deb
}

enum TimeStatus {
    unknown @0;                         # Status of the process is not known. 
    notRunning @1;                      # Time synchronization is currently not running.
    runningWaiting @2;                  # Time synchronization is currently waiting for upstream clock.
    runningSynchronized @3;             # Time synchronization is completed. 
}

enum CameraDriverStatus {
    unknown @0;                         # Process is running but the exact state is unknown.
    waitingForTimeSync @1;              # Process is waiting for time synchronization to complete.
    waitingForHardware @2;              # Process is waiting for hardware to respond to initialization.
    runningWithBuiltinCamera @3;        # Process is running with built-in camera.
    runningWithExternalCamera @4;       # Process is running with external camera.
}

enum VioStatus {
    unknown @0;                         # Process is running but the exact state is unknown.
    initializing @1;                    # Process is initializing.
    running @2;                         # Process is running normally.
}

enum RecordStatus {
    unknown @0;                         # Status of the process is not known. 
    notRunning @1;                      # Recording process is not running.
    notRunningFailed @2;                # Recording process is not running due to a previous failure.
    running @3;                         # Recording process is runnning.
}

enum PlaybackStatus {
    unknown @0;                         # Status of the process is not known. 
    notRunning @1;                      # Recording process is not running.
    notRunningFailed @2;                # Recording process is not running due to a previous failure.
    running @3;                         # Recording process is runnning.
}

enum ScriptStatus {
    unknown @0;                         # Status of the process is not known. 
    terminatedSuccess @1;               # Script process has terminated in success.
    terminatedFailure @2;               # Script process has terminated in failure.
    running @3;                         # Script process is currently running.
}

enum ViewerStatus {
    unknown @0;                         # Status of the process is not known. 
    notRunning @1;                      # Recording process is not running.
    notRunningFailed @2;                # Recording process is not running due to a previous failure.
    running @3;                         # Recording process is runnning.
}

enum ReprojectionStatus {
    unknown @0;                         # Status of the process is not known.
    notRunning @1;                      # Reprojection runner is not running.
    notRunningFailed @2;                # Reprojection runner is not running due to a previous failure.
    running @3;                         # Reprojection runner is running.
}

struct SensorTemperature {
    name @0: Text;                      # Name of sensor
    temperature @1: Float32;            # Temperature of sensor
}

struct CameraDriverHeartbeat {
    status @0: CameraDriverStatus;      # Status of the camera driver.
    temperature @1: Float32;            # Temperature of the VPU.
    configuration @2: Text;             # Configuration that was used to run the camera driver.
    runtime @3: UInt64;                 # Runtime of the process in seconds
    sensorTemperatures @4: List(SensorTemperature);  # List of sensor Temperatures
}

struct VioHeartbeat {
    status @0: VioStatus;               # Status of the VIO.
    temperature @1: Float32;            # Temperature of the VPU.
    configuration @2: Text;             # Configuration that was used to run the VIO.
    runtime @3: UInt64;                 # Runtime of the process in seconds
}

struct ScriptHeartbeat {
    id @0: UInt32;                      # Identifier of the script process.
    name @1: Text;                      # Name of the script that spawned the process.
    status @2: ScriptStatus;            # Status of the script process.
}

struct ScriptLog {
    id @0: UInt32;                      # Identifier of the script process.
    line @1: Text;                      # Single line of log from the script process.
}

struct CpuInfo {
    usage @0: Float32;                  # Percent between 0.0 to 100.0 indicating usage of CPU.
}

struct RamInfo {
    used @0: UInt64;                    # Number of used bytes in the RAM.
    total @1: UInt64;                   # Total number of bytes in the RAM.
}

struct DiskInfo {
    used @0: UInt64;                    # Number of used bytes in the disk.
    total @1: UInt64;                   # Total number of bytes in the disk.
}

struct TemperatureInfo {
    averageCelsius @0: Float32;         # Average temperature of all CPU in the device. 
    maximumCelsius @1: Float32;         # Maximum temperature amongst all CPU in the device.
}

struct NetworkConfiguration {
    identifier @0: UInt8;               # Specifies a unique identifier that will affect the hostname and MAC-address of the device. 
    dhcp @1: Bool;                      # Specifies whether the device will act as a DHCP client. If set to true, the "address", "gateway" and "dns" fields will be ignored.
    address @2: Text;                   # Specifies the IP address of the device.
    gateway @3: Text;                   # Specifies the IP address of the default gateway of the device.
    dns @4: Text;                       # Specifies the IP address of the DNS server to use on the device.
    prefix @5: UInt8;                   # Specifies the network prefix length of the device (number of bits set in the network mask).
}

enum TimeRole {
    default @0;                         # System will synchronize to external NTP time servers.
    master @1;                          # System will use its own clock as the ground truth.
    boundaryPPS @2;                     # System will synchronize its time to a GPS time source (UDP NMEA + PPS).
    slavePPS @3;                        # System will synchronize its own time to either a Master and Boundary node.
    boundarySoftware @4;                # System will synchronize with a grandmaster clock using software PTP(master unit).
    slaveSoftware @5;                   # System will synchronize with a grandmaster clock using software PTP(slave unit).
    boundaryHardware @6;                # System will synchronize with a grandmaster clock on the ethernet port and become a grandmasterclock on 100baseT port.
    slaveHardware @7;                   # System will synchronize with the grandmasterclock port on a boundaryHardware unit.
}

enum CompressionAlgorithm {
    none @0;                            # No compression.
    lz4 @1;                             # LZ4 algorithm.
    zstd @2;                            # ZSTD algorithm.
}

enum CompressionLevel {
    fastest @0;                         # Fastest speed but least compression done.
    fast @1;                            # Fast speed but less compression done.
    default @2;                         # Default speed and default compression done.
    slow @3;                            # Slow speed but more compression done.
    slowest @4;                         # Slowest speed but most compression done.
}

enum ViewerDataMode {
    none @0;                            # Regular topics     
    lowerData @1;                       # Only view odometry, flow, 2d detections, vio_state
    lowestData @2;                      # Only view odometry, 2d detections
}

struct ExecuteScript {
    script @0: Text;                    # Specifes the name of the script to execute.
    arguments @1: List(Text);           # Specifies the arguments to be used when executing the script.
}

struct UploadUpdateBegin {
    fileName  @0 :Text;                 # Normalized file name (no directories)
    totalSize @1 :UInt64;               # Full number of bytes expected across all chunks
}

struct UploadUpdateChunk {
    fileName  @0 :Text;                 # Target file being streamed
    offset    @1 :UInt64;               # Byte position for this chunk; ensures ordered writes
    data      @2 :Data;                 # Raw payload for this chunk
}

struct UploadUpdateFinish {
    fileName  @0 :Text;                 # Signals completion for the given file
}
