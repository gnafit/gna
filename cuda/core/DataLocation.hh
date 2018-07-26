#ifndef GNACUDATALOCATION_H
#define GNACUDATALOCATION_H

#include <iostream>


enum class SyncFlag {
	Unsynchronized = 0,
	Synchronized,
	SyncFailed
};

inline std::ostream& operator<<(std::ostream& so, SyncFlag sFlag) {
        switch (sFlag) {
                case SyncFlag::Synchronized:
                        so << "Synchronized";
                        break;
                case SyncFlag::Unsynchronized:
                        so << "Unsynchronized";
                        break;
                case SyncFlag::SyncFailed:
                        so << "SyncFailed";
                        break;
                default:
                        so.setstate(std::ios_base::failbit);
        }
        return so;
}



enum class DataLocation {
	NotInitialized = 0,
        InitializedOnly,
        Host,
        Device,
	NoData, 
	Crashed
};

inline std::ostream& operator<<(std::ostream& so, DataLocation gState) {
        switch (gState) {
                case DataLocation::NotInitialized:
                        so << "NotInitialized";
                        break;
                case DataLocation::InitializedOnly:
                        so << "InitializedOnly";
                        break;
                case DataLocation::Host:
                        so << "Host";
                        break;
                case DataLocation::Device:
                        so << "Device";
                        break;
                case DataLocation::NoData:
                        so << "NoData";
                        break;
                case DataLocation::Crashed:
                        so << "Crashed";
                        break;
                default:
                        so.setstate(std::ios_base::failbit);
        }
        return so;
}


#endif /* GNACUDATALOCATION_H */
