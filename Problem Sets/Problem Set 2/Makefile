NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/local/lib:/usr/local/cuda-7.0/lib64
OPENCV_INCLUDEPATH=/usr/local/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-7.0/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-O3 -arch=sm_30 -Xcompiler -Wall -Xcompiler -Wextra -m64 -DCUDA_ARCH_BIN="3.5"

GCC_OPTS=-O3 -Wall -Wextra -m64

$(eval OPENCV_OPTS=$(shell pkg-config opencv --cflags --libs;echo $))

student: main.o student_func.o compare.o reference_calc.o Makefile
	$(NVCC) -o HW2 main.o student_func.o compare.o reference_calc.o -L $(OPENCV_OPTS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h HW2.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(OPENCV_OPTS) -I $(CUDA_INCLUDEPATH)

student_func.o: student_func.cu reference_calc.cpp utils.h
	nvcc -c student_func.cu $(NVCC_OPTS)

compare.o: compare.cpp compare.h
	g++ -c compare.cpp -I $(OPENCV_OPTS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_OPTS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o *.png hw
