#  potree to cesium 3dtiles
## Thanks to Potree and cesium for their contributions for rendering large point clouds
>> https://github.com/potree  
>> https://github.com/potree/PotreeConverter  
>> https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/specification


# usage:
## potreeConvert 1.x version,test 1.7
```
# generate compressed LAZ files instead of the default BIN format.
./PotreeConverter.exe C:/data.las -o C:/potree_converted --output-format LAS -p pageName
```

## potreeConvert 2.x version,test 2.0
```
Optionally specify the sampling strategy:
Poisson-disk sampling (default): 
PotreeConverter.exe <input> -o <outputDir> -m poisson
Random sampling: 
PotreeConverter.exe <input> -o <outputDir> -m random
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
        viewer.zoomTo(tileset);
    }
    
```

## todos
   - [x] support potreeconvert 2.0
   - [x] support las to 3dtiles,**now only support for windows**
   - [ ] support save to db files(ex:sqlite)
   - [ ] merge multi 3dtiles 
   - [ ] speed ​​up data convert 