echo day,b1,b3,b7
for name in $@; do
    echo Processing $name 1>&2
    sds=HDF4_EOS:EOS_SWATH:\"$name\":MODIS_SWATH_Type_L1B:EV_1KM_RefSB
    mosaic=$name.tif
    warp=$name.warp.tif
    gdal_translate -b 1 -b 3 -b 7 $sds $mosaic > /dev/null
    gdalwarp -t_srs epsg:4326 $mosaic $warp > /dev/null
    vals=$(gdallocationinfo $warp -valonly -wgs84 116.6031 40.0799 | grep -v 65535)
    if [[ $vals ]]; then
	echo ${name:10:7} $vals | tr ' ' ','
    fi
done
