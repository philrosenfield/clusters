cut -d , -f 7,10,11 $1 > $2
# if HST formatting, column 10 will be Dataset, not RA. U denotes WFPC2, J is ACS, I is WFC3
# see http://archive.stsci.edu/search_fields.php?mission=hst
for f in $(grep ,U $2 | cut -d , -f 2)
do
dataset=$(grep $f $1 | cut -d , -f 10-11)
radec=$(grep $f $1 | cut -d , -f 11-12)
sed -i ' ' "s/$dataset/$radec/" $2

# remove MAST header formatting string,string etc...
sed -i ' ' '/^ string/ d' $2

# not sure why bash adds a space...
mv ${2}\  $2
done
