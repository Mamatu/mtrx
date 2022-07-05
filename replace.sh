sed -i "s/\n//g" $1
sed -i "s/, ValueType .*,/,/g" $1
sed -i "s/, ValueType .*,/,/g" $1
sed -i "s/,\s*ValueType.*,/,/g" $1
sed -i "s/,\s*ValueType.*)/)/g" $1
sed -i 's/\s*ValueType [a-z]*Type)/)/g' $1
sed -i 's/\s*ValueType [a-z]*Type,/,/g' $1 
sed -i "s/Mem \*/T */g" $1
sed -i "s/void \*/T */g" $1
sed -i "s/^:Mems/typename BlasGeneric<T>::Mems/g" $1
./code_format.sh
