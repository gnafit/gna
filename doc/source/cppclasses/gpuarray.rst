GNAcuGpuArray
~~~~~~~~~~~~~

Description
^^^^^^^^^^^

Template wrapper class for GPU access allows to perfom vector (vectorized matrix) operations. It contains size of array, device and host pointer on it. Host pointer is uninitialized by default. Data device-to-host transfer accurs on demand. It also contains memory state flag to track data placement and errors.

Typename may be double, float or int.

Constructors
^^^^^^^^^^^^

``GNAcuGpuArray()``

``GNAcuGpuArray(size_t inSize)``

where `inSize` is an array size.

Initializer
^^^^^^^^^^^

``DataLocation Init(size_t inSize)``

The same as constructor. Returns `Device` state in case of successful initialization and `Crashed` in case of device memory allocation error.

Member functions
^^^^^^^^^^^^^^^^

``void resize(size_t newSize)`` reallocates memory with a new size in the same class instance.

``void setSize(size_t inSize)`` sets size member value.

``DataLocation setByHostArray(T* inHostArr)`` copies data from host input array to current class instance device memory. 

``DataLocation setByDeviceArray(T* inDeviceArr)`` copies data from device input array to current class instance device memory.

``DataLocation setByValue(T value)`` sets current device array by input value.

``DataLocation getContentToCPU(T* dst)`` copies data from current device memory to host-placed `dst`.

``DataLocation getContent(T* dst)`` copies data from current instance device memory to device-placed `dst`.

``DataLocation transferH2D()`` copies data from current instance host memory to device memory.

``void transferD2H()`` copies data from current instance device memory to host memory.

``T* getArrayPtr()`` returns device pointer.
 
``void setArrayPtr(T* inDevPtr)`` sets device pointer.

``size_t getArraySize()`` returns array size.

``void negate()`` changes array values to inverse values. 

``void dump()`` prints array values.


Operators
---------

``GNAcuGpuArray<T>& operator+=(GNAcuGpuArray<T> &rhs)``

``GNAcuGpuArray<T>& operator-=(GNAcuGpuArray<T> &rhs)``

``GNAcuGpuArray<T>& operator*=(GNAcuGpuArray<T> &rhs)``

``GNAcuGpuArray<T>& operator*=(T rhs)``

``GNAcuGpuArray<T> operator=(GNAcuGpuArray<T> rhs)``
