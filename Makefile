TARGET = xtest

.PHONY: all clean

all:
	$(CC) 20260222-SubMixBS.c -o $(TARGET) -Ofast -march=native
#	$(CC) 20260222-SubMixBS.c -o $(TARGET) -O -march=native
#	./$(TARGET)

clean:
	$(RM) $(TARGET)
