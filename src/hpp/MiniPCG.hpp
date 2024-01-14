// #pragma once
// #include <cstdint>

namespace Charlie {


// From https://www.pcg-random.org/download.html
// and https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.h
// PCG that generates 32-bit unsigned integers.
struct MiniPCG
{
  
  typedef uint32_t result_type; // DO NOT DELETE! WILL BE USED IN std DISTRIBUTION CLASS.
  std::uint64_t state, inc;
  MiniPCG() { state = 0x853c49e6748fea9bULL - 42ULL; inc = 0xda3e39cb94b95bdbULL; }
  void seed(std::uint64_t s)
  {
    state = s + 0x853c49e6748fea9bULL - 42ULL; inc = 0xda3e39cb94b95bdbULL + s;
  }
  MiniPCG(std::uint64_t s) { seed(s); }
  
  
  constexpr const static std::uint32_t min() { return 0ul; }
  constexpr const static std::uint32_t max() { return 4294967295ul; } 


  std::uint32_t operator()()
  {
    std::uint64_t oldstate = this->state;
    // Advance internal state
    this->state = oldstate * 6364136223846793005ULL + (this->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    std::uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    std::uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }
};


}
