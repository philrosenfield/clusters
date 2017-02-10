# Get HST Datasets to download!

1. Identify possible fields.
   * On discovery portal, narrow by SFH filters and radius
     * download full data table (all columns, including `s_region`)
   * `filename: MAST_...csv`
2. Cross match with known clusters.
   * Use clusters.footprints.crossmatch with literature table
   (if not Bica2008, check hard coded column names)
   * `filename: MAST_...matched_Bica2008.csv`
   * `$ python -m clusters.footprints.cross_match littable MASTtable`
4. Cull cross matched table for input to HST to identify data sets to retrieve.
   *Use tradec.sh to cull matched csv to target, ra, dec to upload to hst search
   * `$ tradec.sh inputfile outputfile`
   * (must specify outputfile) `filename: fileupload_...csv`
5. [4] result has multiple entries (for each pointing/filter).
   * Use `uniqify.py` (on RA) to make a culled file.
   * `filename: fileupload_...unique_ra.csv`
   * ``$ python uniqify.py infile --column=s_ra` (or RAJ2000 or whatever it is)
6. Upload [5] result to [hst search](http://archive.stsci.edu/hst/search.php?form=fuf)
   * Mark appropriate delimiter, column #, de-select Resolver
      NOTE: May need to iterate the search radius: likely <= 1'
   *    Select imagers: WFPC2
   *    Filters/Gratings:
      F336W,F225W,F275W,F555W,F439W,F450W,F606W,F814W
        if you miss this, will need to do the following on the result:
          `grep -E "F336W|F225W|F275W|F555W|F439W|F450W|F606W|F814W"`
   *    Search output format: csv
   *    Max record for target: max
   *    Select degrees for units
7. Download the file, varify, format, and uniqify.
   * $ grep Warnings and no rows found in [7]
   (typically, header row or RA being Dataset see source tradec.sh)
   * Check that the final number of `Input line` in [7]
      matches the number of lines in [5]
   * remove Input lines comments
   `$ bash rmInputline.sh [6]`
   * Edit the [7] header to not have spaces and parentheses
   * Multiple datasets for each filter/pointing. So uniqify
   $ `python uniqify.py [7] --column=Dataset`
8. Upload [7] to Discovery Portal and check coverage against [2].
   * first line of "Generic Table" should have something starting with #@
      and the second line is the (uncommented) header.
   e.g.:
   `#@string,string,ra,dec,integer,datetime,datetime,float,string,string,string,float,integer,datetime,string,integer,float
   Dataset,Target Name,RAJ2000,DEJ2000,Ref,Start Time,Stop Time,Exp Time,Instrument,Apertures,Filters/Gratings,Central Wavelength,Proposal ID,Release Date,Preview Name,High-Level Science Products,Ang Sep (')`
   * Repeat 6a-8 until everything matches.
9. Cut [8] to first column (Dataset)
   * `$ python uniqify.py [8] --column=Dataset`
   * `$ cut_by_dataset.sh [8] > [9]`
10. Upload [9] or paste its contents into
   [hst dataset lookup ](http://archive.stsci.edu/cgi-bin/dataset_lookup)
11. Decide whether or not to uncheck unnecessary filters or just download them all
   * cutfilters.sh (in downloaded directory)
12. Make a MASTlike table of the downloaded files
   * `$ python clusters.data_wrangling.mastlike_table *drz*fits`
13. Cross match [12] with [2] adding Dataset (actual downloaded dataset)
   * `$ python -m clusters.footprints.cross_match [12] [1] --ra s_ra --dec s_dec --namecol Dataset`
   * rename [13] final_table.csv
