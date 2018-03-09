new_area=$1
mkdir $new_area
cp template.tex $new_area/$new_area.tex
cp template.bib $new_area/$new_area.bib

cd $new_area
# sed -i s/template/$new_area/g $new_area.tex
sed -i s/bibliography\{template\}/bibliography\{$new_area\}/ $new_area.tex
echo "bla"
echo $new_area
title=`echo $new_area | tr '_' ' ' `
echo "meme"
echo $title
sed -i s/title\{template\}/title\{$(title)\}/ $new_area.tex

pdflatex $new_area.tex && bibtex $new_area && pdflatex $new_area.tex &&
  pdflatex $new_area.tex && evince $new_area.pdf
