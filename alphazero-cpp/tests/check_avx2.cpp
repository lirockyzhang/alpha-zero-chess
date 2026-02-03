// Quick test to check if AVX2 is enabled during compilation
#include <iostream>

int main() {
#if defined(__AVX2__)
    std::cout << "AVX2 is ENABLED (__AVX2__ is defined)" << std::endl;
    return 0;
#else
    std::cout << "AVX2 is NOT enabled (__AVX2__ is NOT defined)" << std::endl;

    #if defined(_M_X64) || defined(_M_AMD64)
        std::cout << "  - Platform: x64 detected" << std::endl;
    #endif

    #if defined(__AVX__)
        std::cout << "  - AVX (256-bit) is available" << std::endl;
    #else
        std::cout << "  - AVX is also NOT available" << std::endl;
    #endif

    #if defined(_MSC_VER)
        std::cout << "  - Compiler: MSVC version " << _MSC_VER << std::endl;
        std::cout << "  - Hint: Add /arch:AVX2 to enable AVX2" << std::endl;
    #elif defined(__GNUC__)
        std::cout << "  - Compiler: GCC/Clang" << std::endl;
        std::cout << "  - Hint: Add -mavx2 to enable AVX2" << std::endl;
    #endif

    return 1;
#endif
}
