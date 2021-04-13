#  potree to cesium 3dtiles
## Thanks to Potree and cesium for their contributions for rendering large point clouds
>> https://github.com/potree  
>> https://github.com/potree/PotreeConverter  
>> https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/specification



# usage:
## potreeConvert
```
# generate compressed LAZ files instead of the default BIN format.
./PotreeConverter.exe C:/data.las -o C:/potree_converted --output-format LAS -p pageName
```

## potree23dtiles
see function for specific usage. 
```
convert23dtiles(src,outdir,proj_param,max_level=5) 
>> src:potree out data dir,include cloud.js file
>> outdir:output
>> proj_param:proj4 param,ex:EPSG:32650
>> max_level:max tree node level,default 15
```

## cesium
```
    var tileset = new Cesium.Cesium3DTileset({ url: "http://127.0.0.1/test/tileset.json" });
   
    tileset.readyPromise.then(function(data) {
        viewer.scene.primitives.add(data);
    }

```

## todos
   - [ ] support potreeconvert 2.0
   - [ ] support save to db files(ex:sqlite)
   - [ ] support las to 3dtiles 
   - [ ] merge multi 3dtiles 