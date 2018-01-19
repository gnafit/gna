#ifndef GNACUDATALOCATION_H
#define GNACUDATALOCATION_H

#include <iostream>


enum SyncFlag {
	Synchronized,
	Unsinchronized,
	SyncFailed
}

inline std::ostream& operator<<(std::ostream& so, SyncFlag sFlag) {
        switch (sFlag) {
                case Synchronized:
                        so << "Synchronized";
                        break;
                case Unsinchronized:
                        so << "Unsinchronized";
                        break;
                case SyncFailed:
                        so << "SyncFailed";
                        break;
                default:
                        so.setstate(std::ios_base::failbit);
        }
        return so;
}



enum DataLocation {
	NotInitialized,
        InitializedOnly,
        Host,
        Device,
	NoData, 
	Crashed
};

inline std::ostream& operator<<(std::ostream& so, DataLocation gState) {
        switch (gState) {
                case NotInitialized:
                        so << "NotInitialized";
                        break;
                case InitializedOnly:
                        so << "InitializedOnly";
                        break;
                case Host:
                        so << "Host";
                        break;
                case Device:
                        so << "Device";
                        break;
                case NoData:
                        so << "NoData";
                        break;
                case Crashed:
                        so << "Crashed";
                        break;
                default:
                        so.setstate(std::ios_base::failbit);
        }
        return so;
}


#endif /* GNACUDATALOCATION_H */
