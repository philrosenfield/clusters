python -m clusters.data_wrangling.print_key FILTNAM1 *drz* > filternames.txt
grep -vE "F336W|F225W|F275W|F555W|F439W|F450W|F606W|F814W" filternames.txt > badfilts.txt
mkdir nonsfh
cut -d ' ' -f 1 badfilts.txt | cut -d . -f 1 > mvlist.sh
sed -i ' ' 's/^/mv /' mvlist.sh
sed -i ' ' 's/$/* nonsfh/' mvlist.sh
bash mvlist.sh

