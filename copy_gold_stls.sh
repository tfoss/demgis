#!/bin/bash
# Copy all correct/recommended STLs to GOLD_STLs directory
# Based on EURASIA_STL_RECOMMENDED.md
#
# IMPORTANT: When generating new/fixed STLs, ALWAYS copy them to GOLD_STLs immediately:
#   cp <source_dir>/<country>_*.stl GOLD_STLs/<region>/
#
# This ensures the GOLD_STLs directory stays up-to-date with latest corrections.

set -e

echo "Copying recommended STLs to GOLD_STLs directory..."
echo ""

# CoastalFix countries (29 total) - corrected star types
COASTAL_FIX_DIR="STLs_Eurasia_CoastalFix_20260114_144757_753fc5f"

echo "Copying CoastalFix countries (29)..."

# Europe (13)
for country in Albania Belgium Bulgaria Croatia France Germany Italy Lithuania Montenegro Poland Romania Spain Ukraine; do
    cp "$COASTAL_FIX_DIR/Europe/${country}_solid.stl" GOLD_STLs/Europe/
done

# Middle East (6) - Note: Turkey will be overridden with merged version
for country in Cyprus Egypt Israel "Saudi Arabia" Syria Turkey; do
    if [ "$country" = "Saudi Arabia" ]; then
        cp "$COASTAL_FIX_DIR/MiddleEast/Saudi_Arabia_solid.stl" GOLD_STLs/MiddleEast/
    else
        cp "$COASTAL_FIX_DIR/MiddleEast/${country}_solid.stl" GOLD_STLs/MiddleEast/
    fi
done

# South Asia (3)
for country in Bangladesh India Pakistan; do
    cp "$COASTAL_FIX_DIR/SouthAsia/${country}_solid.stl" GOLD_STLs/SouthAsia/
done

# Southeast Asia (3)
for country in Cambodia Myanmar Vietnam; do
    cp "$COASTAL_FIX_DIR/SoutheastAsia/${country}_solid.stl" GOLD_STLs/SoutheastAsia/
done

# East Asia (4)
for country in China "North Korea" "South Korea" Taiwan; do
    if [ "$country" = "North Korea" ]; then
        cp "$COASTAL_FIX_DIR/EastAsia/North_Korea_solid.stl" GOLD_STLs/EastAsia/
    elif [ "$country" = "South Korea" ]; then
        cp "$COASTAL_FIX_DIR/EastAsia/South_Korea_solid.stl" GOLD_STLs/EastAsia/
    else
        cp "$COASTAL_FIX_DIR/EastAsia/${country}_solid.stl" GOLD_STLs/EastAsia/
    fi
done

echo "  ✓ Copied 29 CoastalFix countries"
echo ""

# Individual fixes
echo "Copying individual fixes (8)..."

# Yemen
cp "STLs_Eurasia_YemenFix_20260115_083538_4e63a67/MiddleEast/Yemen_solid.stl" GOLD_STLs/MiddleEast/

# Azerbaijan
cp "STLs_Eurasia_AzerbaijanStarup_20260115_090730_824103b/Caucasus/Azerbaijan_starup.stl" GOLD_STLs/Caucasus/

# Luxembourg (custom generation with reduced smoothing for very small country)
cp "STLs_Luxembourg_20260118_134403_e20a0b8/Luxembourg_solid.stl" GOLD_STLs/Europe/

# Denmark (3-island connection: Jutland + Zealand + Funen, excludes Bornholm)
# Physical low bridges (1.5mm height) with smoothed coastlines (0.01° buffer) for wide attachment zones
cp "STLs_Denmark_Connected_20260122_122332_e20a0b8/Denmark_starup.stl" GOLD_STLs/Europe/

# Thailand (changed from extruded to cut star - Bangkok is inland)
cp "STLs_Denmark_Thailand_Fix_20260121_083857_e20a0b8/SoutheastAsia/Thailand_solid.stl" GOLD_STLs/SoutheastAsia/

# Sri Lanka (extended DEM coverage to 5°N for complete southern region)
cp "STLs_SriLanka_Iceland_20260121_120545_e20a0b8/SouthAsia/Sri_Lanka_starup.stl" GOLD_STLs/SouthAsia/

# Kosovo (newly added country - was missing from initial generation)
cp "STLs_Iceland_Kosovo_20260122_073114_e20a0b8/Europe/Kosovo_solid.stl" GOLD_STLs/Europe/

# Iceland (separate DEM with Iceland-centered AEA projection)
cp "STLs_Iceland_20260122_083507_e20a0b8/Iceland_starup.stl" GOLD_STLs/Europe/

# Turkey merged version not yet generated - using CoastalFix version (correct star type)
# Note: Turkey merge fix is in code but regeneration output not persisted

# Note: Kazakhstan, Serbia, Czechia, Turkey merge will come from Full/CoastalFix directories
# (fixes were applied to make_eurasia_all.py but regeneration outputs not persisted)

echo "  ✓ Copied 8 individual fixes (Yemen, Azerbaijan, Luxembourg, Denmark, Thailand, Sri Lanka, Kosovo, Iceland)"
echo ""

# Full generation countries (49 total - includes Kazakhstan, Serbia, Czechia)
# Note: Fixes for these 3 are in code but regeneration outputs not persisted
FULL_DIR="STLs_Eurasia_Full_20260113_210127_5c16017"

echo "Copying Full generation countries (49 including KZ/RS/CZ)..."

# Europe (19) - Note: Czechia, Serbia missing from Full (name mismatches, need regeneration)
for country in Austria Belarus "Bosnia and Herzegovina" Denmark Estonia Finland Greece Hungary Ireland Latvia Moldova Netherlands "North Macedonia" Norway Portugal Russia Slovakia Slovenia Sweden Switzerland "United Kingdom"; do
    case "$country" in
        "Bosnia and Herzegovina")
            # Try both _starup and _solid
            if [ -f "$FULL_DIR/Europe/Bosnia_and_Herzegovina_starup.stl" ]; then
                cp "$FULL_DIR/Europe/Bosnia_and_Herzegovina_starup.stl" GOLD_STLs/Europe/
            elif [ -f "$FULL_DIR/Europe/Bosnia_and_Herzegovina_solid.stl" ]; then
                cp "$FULL_DIR/Europe/Bosnia_and_Herzegovina_solid.stl" GOLD_STLs/Europe/
            fi
            ;;
        "North Macedonia")
            if [ -f "$FULL_DIR/Europe/North_Macedonia_solid.stl" ]; then
                cp "$FULL_DIR/Europe/North_Macedonia_solid.stl" GOLD_STLs/Europe/
            elif [ -f "$FULL_DIR/Europe/North_Macedonia_starup.stl" ]; then
                cp "$FULL_DIR/Europe/North_Macedonia_starup.stl" GOLD_STLs/Europe/
            fi
            ;;
        "United Kingdom")
            if [ -f "$FULL_DIR/Europe/United_Kingdom_starup.stl" ]; then
                cp "$FULL_DIR/Europe/United_Kingdom_starup.stl" GOLD_STLs/Europe/
            elif [ -f "$FULL_DIR/Europe/United_Kingdom_solid.stl" ]; then
                cp "$FULL_DIR/Europe/United_Kingdom_solid.stl" GOLD_STLs/Europe/
            fi
            ;;
        *)
            # Try both _starup and _solid (use whatever exists)
            if [ -f "$FULL_DIR/Europe/${country}_starup.stl" ]; then
                cp "$FULL_DIR/Europe/${country}_starup.stl" GOLD_STLs/Europe/
            elif [ -f "$FULL_DIR/Europe/${country}_solid.stl" ]; then
                cp "$FULL_DIR/Europe/${country}_solid.stl" GOLD_STLs/Europe/
            fi
            ;;
    esac
done

# Middle East (9) - excluding Yemen (individual fix)
for country in Iran Iraq Jordan Kuwait Lebanon Oman Palestine Qatar "United Arab Emirates"; do
    if [ "$country" = "United Arab Emirates" ]; then
        cp "$FULL_DIR/MiddleEast/United_Arab_Emirates_starup.stl" GOLD_STLs/MiddleEast/
    else
        # Try both _starup and _solid
        if [ -f "$FULL_DIR/MiddleEast/${country}_starup.stl" ]; then
            cp "$FULL_DIR/MiddleEast/${country}_starup.stl" GOLD_STLs/MiddleEast/
        elif [ -f "$FULL_DIR/MiddleEast/${country}_solid.stl" ]; then
            cp "$FULL_DIR/MiddleEast/${country}_solid.stl" GOLD_STLs/MiddleEast/
        fi
    fi
done

# Caucasus (2) - excluding Azerbaijan (individual fix)
for country in Armenia Georgia; do
    if [ -f "$FULL_DIR/Caucasus/${country}_solid.stl" ]; then
        cp "$FULL_DIR/Caucasus/${country}_solid.stl" GOLD_STLs/Caucasus/
    elif [ -f "$FULL_DIR/Caucasus/${country}_starup.stl" ]; then
        cp "$FULL_DIR/Caucasus/${country}_starup.stl" GOLD_STLs/Caucasus/
    fi
done

# Central Asia (6) - includes Kazakhstan from Full (Baikonur fix in code)
for country in Afghanistan Kazakhstan Kyrgyzstan Tajikistan Turkmenistan Uzbekistan; do
    if [ -f "$FULL_DIR/CentralAsia/${country}_solid.stl" ]; then
        cp "$FULL_DIR/CentralAsia/${country}_solid.stl" GOLD_STLs/CentralAsia/
    elif [ -f "$FULL_DIR/CentralAsia/${country}_starup.stl" ]; then
        cp "$FULL_DIR/CentralAsia/${country}_starup.stl" GOLD_STLs/CentralAsia/
    fi
done

# South Asia (3)
for country in Bhutan Nepal "Sri Lanka"; do
    if [ "$country" = "Sri Lanka" ]; then
        cp "$FULL_DIR/SouthAsia/Sri_Lanka_starup.stl" GOLD_STLs/SouthAsia/
    else
        if [ -f "$FULL_DIR/SouthAsia/${country}_solid.stl" ]; then
            cp "$FULL_DIR/SouthAsia/${country}_solid.stl" GOLD_STLs/SouthAsia/
        elif [ -f "$FULL_DIR/SouthAsia/${country}_starup.stl" ]; then
            cp "$FULL_DIR/SouthAsia/${country}_starup.stl" GOLD_STLs/SouthAsia/
        fi
    fi
done

# Southeast Asia (3) - Malaysia failed
for country in Laos Philippines Thailand; do
    if [ -f "$FULL_DIR/SoutheastAsia/${country}_starup.stl" ]; then
        cp "$FULL_DIR/SoutheastAsia/${country}_starup.stl" GOLD_STLs/SoutheastAsia/
    elif [ -f "$FULL_DIR/SoutheastAsia/${country}_solid.stl" ]; then
        cp "$FULL_DIR/SoutheastAsia/${country}_solid.stl" GOLD_STLs/SoutheastAsia/
    fi
done

# East Asia (2)
for country in Japan Mongolia; do
    if [ -f "$FULL_DIR/EastAsia/${country}_starup.stl" ]; then
        cp "$FULL_DIR/EastAsia/${country}_starup.stl" GOLD_STLs/EastAsia/
    elif [ -f "$FULL_DIR/EastAsia/${country}_solid.stl" ]; then
        cp "$FULL_DIR/EastAsia/${country}_solid.stl" GOLD_STLs/EastAsia/
    fi
done

echo "  ✓ Copied 46 Full generation countries"
echo ""

# Separate regions
echo "Copying separate region STLs (2)..."

# Kaliningrad
cp "STLs_Kaliningrad_20260115_114401_6eab876/Kaliningrad_starup.stl" GOLD_STLs/Europe/

# Northern Ireland
cp "STLs_NorthernIreland_20260116_083547_caa45ed/Northern_Ireland_starup.stl" GOLD_STLs/Europe/

echo "  ✓ Copied 2 separate regions"
echo ""

# Count total files
echo "================================================"
echo "GOLD_STLs Summary:"
echo "================================================"
for region in Europe MiddleEast Caucasus CentralAsia SouthAsia SoutheastAsia EastAsia; do
    count=$(ls -1 GOLD_STLs/$region/*.stl 2>/dev/null | wc -l)
    echo "  $region: $count STLs"
done

total=$(find GOLD_STLs -name "*.stl" | wc -l)
echo "  ----------------"
echo "  TOTAL: $total STLs"
echo ""
echo "✓ All recommended STLs copied to GOLD_STLs/"
