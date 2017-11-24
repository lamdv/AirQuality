progname=$(basename $0)

usage() {
    echo "Usage:" 1>&2
    echo "    sh $progname query.csv" 1>&2
    echo "Where query.csv is the LAADS query result." 1>&2
    echo "Resuming an interrupted download is supported." 1>&2
}

download() {
    cut -d, -f2 $1 | tail -n +2 |
	wget --continue \
	     --base=https://ladsweb.modaps.eosdis.nasa.gov \
	     --directory-prefix=data \
	     --input-file=-
}

case $# in
    1) download $1 ;;
    *) usage; exit 1 ;;
esac
