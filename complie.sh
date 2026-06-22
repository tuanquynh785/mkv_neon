# Biên dịch file main.c và file assembly cùng nhau

#rm mkv-neonbs-core
# gcc -O3 -march=armv8-a -o mkv_test mkv-neonbs-glue.c mkv-neonbs-core.S
# aarch64-linux-gnu-gcc mkv-neonbs-glue.c mkv-neonbs-core.S -o mkv_test -static
aarch64-linux-gnu-gcc -O3 -march=armv8-a+simd -static -o mkv-neonbs-core mkv-neonbs-glue.c mkv-neonbs-core.S
# arm-linux-gnueabihf mkv-neonbs-glue.c mkv-neonbs-core.S -o mkv_test -static
# Chạy thử
qemu-aarch64 ./mkv-neonbs-core
