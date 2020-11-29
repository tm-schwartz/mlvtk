git status > gs.txt
awk '/(modified)/{next} /^[\t\.\.]/{print $1}' gs.txt >> .gitignore  
rm gs.txt
