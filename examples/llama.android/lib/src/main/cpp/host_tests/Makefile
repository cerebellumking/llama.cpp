CXX ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Wpedantic -O0 -g
SANITIZERS ?= -fsanitize=address,undefined
INCLUDES = -I..

test_heterospec_hotpath: test_heterospec_hotpath.cpp ../heterospec_hotpath.hpp
	$(CXX) $(CXXFLAGS) $(SANITIZERS) $(INCLUDES) test_heterospec_hotpath.cpp -o $@
	./$@

clean:
	rm -f test_heterospec_hotpath
