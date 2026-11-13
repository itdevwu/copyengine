# cmake -DINPUT=<bin> -DOUTPUT=<c file> -DNAME=<symbol> -P embed.cmake
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" n)
math(EXPR size "${n} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," body "${hex}")
string(REGEX REPLACE "((0x..,){32})" "\\1\n" body "${body}")
file(WRITE "${OUTPUT}"
  "/* generated from ${INPUT} */\n"
  "__attribute__((aligned(16))) const unsigned char ${NAME}[] = {\n${body}\n};\n"
  "const unsigned long ${NAME}_size = ${size};\n")
