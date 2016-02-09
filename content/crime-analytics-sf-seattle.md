Title: Analyzing 2 months of real crime data from San Francisco and Seattle
Date: 2016-02-05 12:35
Tags: python, pandas, gis, dataviz, data-science, clustering

> The full data analysis described in this blog post can be found in the IPython notebook in the 
[crime-analytics](https://github.com/cmenguy/crime-analytics) repository.

I took part a couple weeks ago in the Coursera course [Communicating Data Science Results](https://www.coursera.org/learn/data-results)
as part of the [Data Science at Scale](https://www.coursera.org/specializations/data-science) specialization.

It has been a great course - one of the assignments in particular was great because it was essentially just about taking
a couple datasets, and coming up with our own problem and solution based on the data.

The datasets in questions proposed were real crime data from [San Francisco](https://data.sfgov.org/Public-Safety/SFPD-Incidents-from-1-January-2003/tmnf-yvry) 
and [Seattle](https://data.seattle.gov/Public-Safety/Seattle-Police-Department-Police-Report-Incident/7ais-f98f).
These being location-based datasets, I decided to take this opportunity to also get more familiar with the Python
geospatial libraries such as [GDAL](http://www.gdal.org/) and how it can be used in tandem with [pandas](http://pandas.pydata.org/) `DataFrames`.

# Temporal Analysis

Before looking at the geospatial distribution, I wanted to narrow it down to the crimes happening during the night since
cursory analysis revealed that the type of crime has a high correlation with time of day (intuitively too).

Initially I wanted to use [seaborn](https://github.com/mwaskom/seaborn)'s heatmap for that, but the results were not 
aesthetically pleasing, so I decided to write my own heatmap system with `matplotlib`.
For that purpose, here is a function below which creates a grid using simple `matplotlib` primitives:

    :::python
    def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                    gridWidth=1.0):
        plt.close()
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
        ax.axes.tick_params(labelcolor='#999999', labelsize='10')
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position('none')
            axis.set_ticks(ticks)
            axis.label.set_color('#999999')
            if hideLabels: axis.set_ticklabels([])
        plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
        map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
        return fig, ax

Then to create the heatmap, we simply need to use `matplotlib.pyplot.imshow` with `nearest` interpolation. I also chose
a grey-scale colormap to better represent the intensity of crimes at any given hour:

    :::python
    plt.imshow(img_src, interpolation='nearest', aspect='auto', cmap=cm.Greys)

The result shows a heatmap where the intensity is normalized across all hours, where a darker color represents a higher
number of crimes at that hour, and lighter color represents a smaller number of crimes.

![Temporal Distribution of Crimes in SF](/images/crime-temporal-distribution.png)

There are some interesting patterns here, and we can see a few crime categories which seem to be particularly frequent
at night. The main offenders seem to be **assault**, **drunkenness**, **larceny/theft**, **prostitution**, **robbery** 
and **vehicle theft**. These are the categories we'll focus on in order to have a meaningful geospatial visualization
to see what San Francisco's crime scene looks like at night.

# Geospatial Analysis

I had to research a few libraries to use in Python in order to visualize on a map the distribution of crimes.
The complicated part is that I wanted to have access to the neighborhoods information, so I could break crimes down
by neighborhood. Unfortunately, `matplotlib`'s [Basemap](http://matplotlib.org/basemap/) toolkit doesn't provide much
aside from 2D maps. A lot of inspiration for this analysis was drawn from this blog post about [blue plaque in London](http://sensitivecities.com/so-youd-like-to-make-a-map-using-python-EN.html).

To solve it we need a [Shapefile](https://en.wikipedia.org/wiki/Shapefile) containing San Francisco's neighborhoods.
Fortunately, there is one available on the [SF open data portal](https://data.sfgov.org/Geographic-Locations-and-Boundaries/SFFind-Neighborhoods/ejmn-jyk6)
and it can easily be used via the `fiona` module.

    :::python
    shp = fiona.open("/path/to/shapefile.shp")

From that we can easily compute the map boundaries `w` and `h`, and feed those to create a `Basemap` instance.
We need a couple parameters to create our `Basemap`:

* The coordinates where our map should be centered. For San Francisco we use `-122`, `37.7`
* The projection to use - in our case we use the [transverse Mercator projection](http://matplotlib.org/basemap/users/tmerc.html)
which should produce a map with less distortion since we are showing a relatively narrow geographic area. For more choice
in projections, see [the complete list](http://matplotlib.org/basemap/users/mapsetup.html)
* The type of ellipsoid - now this one was a bit obscure to me, not being familiar with GIS, but this is actually nothing
more than a coordinate system. It turns out [WGS84](http://wiki.gis.com/wiki/index.php/WGS84) is the standard for GPS
so it makes sense to use that in our case.

We can put all that together to create a `Basemap` instance below:

    :::python
    m = Basemap(
        projection='tmerc',
        lon_0=-122.,
        lat_0=37.7,
        ellps = 'WGS84',
        llcrnrlon=coords[0] - extra * w,
        llcrnrlat=coords[1] - extra + 0.01 * h,
        urcrnrlon=coords[2] + extra * w,
        urcrnrlat=coords[3] + extra + 0.01 * h,
        lat_ts=0,
        resolution='i',
        suppress_ticks=True
    )

Now that we have our `Basemap`, we can add the content of our Shapefile containing San Francisco's neighborhoods:

    :::python
    m.readshapefile(
        '/path/to/shapefile.shp',
        'SF',
        color='none',
        zorder=2
    )

At that point, we need to start creating polygons for each neighborhood so that we can convert them into patches that
can be represented on maps. Two particular libraries can help us achieve what we need:

* [shapely](http://toblerity.github.com/shapely/) is a library for analyzing and manipulating planar geometric objects in Python. 
In particular, it is useful to create `Polygon` and `Point` objects, which for us maps to neighborhoods and crime occurrences.
* [descartes](https://pypi.python.org/pypi/descartes) is the intermediary between shapely's polygons, and matplotlib's maps.
It can be used to create `PolygonPatch` patches which can be represented on a `Basemap`.

For that purpose we create a `DataFrame` where for each neighborhood we will have a `PolygonPatch` based on `Polygon`
objects extracted from the content of the Shapefile.

    :::python
    df_map = pd.DataFrame({
        'poly': [Polygon(xy) for xy in m.SF],
        'ward_name': [ward['name'] for ward in m.SF_info]})
    df_map['area_m'] = df_map['poly'].map(lambda x: x.area)
    df_map['area_km'] = df_map['area_m'] / 100000
    # Draw neighborhoods with polygons
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
        x,
        fc='#000000',
        ec='#ffffff', lw=.5, alpha=1,
        zorder=4))

Now we're ready to draw the crime occurrences on the map. Since our data contains `Latitude` and `Longitude` columns,
we can easily draw them after making sure they're within the polygon formed by San Francisco's neighborhoods.
For example, to draw all occurrences of vehicle theft:

    :::python
    m.scatter(
        [geom.x for geom in sf_night_vehicle_theft_points],
        [geom.y for geom in sf_night_vehicle_theft_points],
        10, marker='o', lw=.25,
        facecolor='cyan', edgecolor='cyan',
        alpha=0.75, antialiased=True,
        label='Vehicle Theft', zorder=3
    )

And of course we need to draw the neighborhoods polygons on the map as well, in order to visualize the distribution
of crimes in the city. This can be done using a `matplotlib.collections.PatchCollection` which can be created from
the neighborhood patches computed previously.

    :::python
    ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))

The final result is a map of San Francisco with all neighborhood boundaries and crimes represented with different
colors for the 6 categories of crimes that happen mostly at night.

![Geospatial Distribution of crimes in SF](/images/crime-geospatial-distribution.png)

We can get a general feeling that the center of the city (especially the Tenderloin) as well as the south are pretty
agitated at night, while the west area is mostly quiet at night in terms of crimes. But this just gives us a general
picture, when we would like to clearly see for each neighborhood how criminal it is. Of course this is a little biased
towards highly-populated neighborhoods, but would still give a good enough idea where to be careful at night.

Before we can divide neighborhoods in criminality level, we need to compute the crime density per neighborhood. This
can easily be done by identifying, for each crime occurrence, in which neighborhood it occurred, count those, and deduce
the crime density.

    :::python
    df_map['count'] = df_map['poly'].map(lambda x: int(len(filter(prep(x).contains, sf_night_crimes_points))))
    df_map['density_m'] = df_map['count'] / df_map['area_m']
    df_map['density_km'] = df_map['count'] / df_map['area_km']

We can then apply a clustering algorithm to group neighborhoods into N criminality buckets. In our case, since there
are not that many neighborhoods, 5 groups seem good enough. The algorithm we're using here is 
[Jenks natural breaks optimization](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). This is a pretty
common method used in cartography software, and is available as part of the [PySAL](https://pysal.readthedocs.org/en/latest/)
library.

    :::python
    from pysal.esda.mapclassify import Natural_Breaks as nb
    
    breaks = nb(
        df_map[df_map['density_km'].notnull()].density_km.values,
        initial=300,
        k=5)

Once the algorithm converges, we have multiple bins that we need to join back to our original neighborhood patches to
figure out to which bin each neighborhood belongs.

    :::python
    jb = pd.DataFrame({'jenks_bins': breaks.yb}, index=df_map[df_map['density_km'].notnull()].index)
    df_map.drop("jenks_bins", inplace=True, axis=1, errors="ignore")
    df_map = df_map.join(jb)

All that is left at that point is to draw again our `PatchCollection`, with the small trick of using the `set_facecolor`
method to apply a color corresponding to the cluster. We used the `Blues` colormap in this case.

    :::python
    cmap = plt.get_cmap('Blues')
    pc = PatchCollection(df_map['patches'], match_original=True)
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_map['jenks_bins'].values)))
    ax.add_collection(pc)

![Crime density clustering](/images/crime-density-tiling.png)

----

This was a pretty fun analysis, and a good introduction to geospatial analysis in Python with `fiona`, `descartes`,
`shapely`, `Basemap` and `PySAL`. We're barely scratching the surface here by looking at only 2 months worth of data - I
might do a follow-up later with more data since we have daily data since January 2003 which is a lot more crime data
to analyze.