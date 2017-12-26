#ifndef GNACUGPUMEMSTATES_H
#define GNACUGPUMEMSTATES_H

#include <iostream>

enum GpuMemoryState {
        NotInitialized,
        InitializedOnly,
        OnHost,
        OnDevice,
        Crashed
};

inline std::ostream& operator<<(std::ostream& so, GpuMemoryState gState) {
        switch (gState) {
                case NotInitialized:
                        so << "NotInitialized";
                        break;
                case InitializedOnly:
                        so << "InitializedOnly";
                        break;
                case OnHost:
                        so << "OnHost";
                        break;
                case OnDevice:
                        so << "OnDevice";
                        break;
                case Crashed:
                        so << "Crashed";
                        break;
                default:
                        so.setstate(std::ios_base::failbit);
        }
        return so;
}

#endif /* GNACUGPUMEMSTATES_H */
