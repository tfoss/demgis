# Eurasia STLs - Recommended Versions for Printing

Generated: 2026-01-15

## Use These Versions

### Corrected Coastal Capitals (29 countries) - Use CoastalFix versions
**Directory**: `STLs_Eurasia_CoastalFix_20260114_144757_753fc5f/`

These have **corrected star types** (cut holes instead of extruded):

**Europe (13):**
- Albania, Belgium, Bulgaria, Croatia, France, Germany, Italy
- Lithuania, Montenegro, Poland, Romania, Spain, Ukraine

**Middle East (6):**
- Cyprus, Egypt, Israel, Saudi Arabia, Syria, Turkey

**South Asia (3):**
- Bangladesh, India, Pakistan

**Southeast Asia (3):**
- Cambodia, Myanmar, Vietnam

**East Asia (4):**
- China, North Korea, South Korea, Taiwan

---

### Additional Individual Fixes - Use These Updated Versions

**Yemen** (corrected star type):
- **Use**: `STLs_Eurasia_YemenFix_20260115_083538_4e63a67/MiddleEast/Yemen_solid.stl`
- **Reason**: Cut star hole (Sana'a is inland, not coastal)
- **Coverage**: 99.9%

**Azerbaijan** (DEM coverage fix + extruded star):
- **Use**: `STLs_Eurasia_AzerbaijanStarup_20260115_090730_824103b/Caucasus/Azerbaijan_starup.stl`
- **Reason**: Extended Caucasus DEM to include eastern region + extruded star (Baku is coastal, low elevation)
- **Coverage**: 98.1% (was 89.4% - missing eastern part)

---

### Full Generation Countries (46 countries) - Use Full versions
**Directory**: `STLs_Eurasia_Full_20260113_210127_5c16017/`

**Europe (21):**
- Austria, Belarus, Bosnia and Herzegovina, Denmark, Estonia
- Finland, Greece, Hungary, Ireland, Latvia, Moldova
- Netherlands, North Macedonia, Norway, Portugal, Russia
- Slovakia, Slovenia, Sweden, Switzerland, United Kingdom

**Middle East (9):**
- Iran, Iraq, Jordan, Kuwait, Lebanon, Oman
- Palestine, Qatar, United Arab Emirates

**Caucasus (2):**
- Armenia, Georgia

**Central Asia (6):**
- Afghanistan, Kazakhstan, Kyrgyzstan, Tajikistan
- Turkmenistan, Uzbekistan

**South Asia (3):**
- Bhutan, Nepal, Sri Lanka

**Southeast Asia (4):**
- Laos, Philippines, Thailand, [Malaysia - failed]

**East Asia (2):**
- Japan, Mongolia

---

## Star Type Reference

**Cut holes** (inland capitals): Star is a hole cut into the base
**Extruded** (coastal capitals): Star raised 2mm above terrain

**Corrected countries**: Now use cut holes (were incorrectly extruded before)

## Total: 77 countries successfully generated
