
def print_sizes():
    cdef unsigned char uc_max = -1
    print(f"unsigned char: {sizeof(unsigned char)} bytes (max={uc_max})")
    cdef unsigned short us_max = -1
    print(f"unsigned short: {sizeof(unsigned short)} bytes (max={us_max})")
    cdef unsigned int ui_max = -1
    print(f"unsigned int: {sizeof(unsigned int)} bytes (max={ui_max})")
