# Prebid module (vendor example)

This folder contains an **example** Real-Time Data (RTD) submodule you can copy and adapt. It shows how Agentic Audiences signals can be read from browser storage and merged into the global ORTB2 fragment as `user.data`, in line with the OpenRTB community extension [**Agentic Audiences in OpenRTB**](https://github.com/InteractiveAdvertisingBureau/openrtb/blob/main/extensions/community_extensions/agentic-audiences.md).

**Do not ship this copy as-is under the sample names.** Vendors are expected to **rename the module file**, align **submodule / build identifiers** with that name, and set **`gvlid`** to their own Global Vendor List ID.

For the **canonical, maintained module** (build flags, publisher configuration, storage format, and behavior), use the implementation and documentation in the **[Prebid.js](https://github.com/prebid/Prebid.js)** repository—not this repo.

## Initiative context

- [IAB Tech Lab — Agentic Audiences](https://iabtechlab.com/standards/agentic-audiences/)
- [IABTechLab/agentic-audiences](https://github.com/IABTechLab/agentic-audiences)

The upstream reference implementation was introduced in [Prebid.js #14626](https://github.com/prebid/Prebid.js/pull/14626).

## What vendors should change

When you implement your own module from this example:

1. **Rename the module file** (e.g. `yourVendorAgenticRtd.js`) and update imports/paths everywhere it is referenced (including the matching test file name).
2. Set **`MODULE_NAME`** (and any exported submodule name) to a string that matches your RTD data provider name and your branding—keep it consistent with `modules/.submodules.json` and your docs.
3. Replace the **`gvlid`** placeholder with your vendor’s numeric GVL ID (see inline comment in the module source).
4. Register the build name in **`modules/.submodules.json`** (see `integration/submodules.json.snippet` for the pattern—your entry will use **your** module name, not the example’s).
5. Adjust **`DEFAULT_STORAGE_KEY`** / **`params.storageKey`** behavior if your integration uses different storage conventions.
6. Update or replace the unit test file so module paths and names match your fork.

After that, follow **[Prebid.js contributing guidelines](https://github.com/prebid/Prebid.js/blob/master/CONTRIBUTING.md)** if you intend to open a pull request against Prebid.js, and point publishers to **Prebid.org / the Prebid.js repo** for how to build and configure the bundle.

## Layout (this example)

| Path | Role |
| --- | --- |
| `modules/agenticAudienceAdapter.js` | Example RTD submodule—**rename and edit** for your vendor module |
| `test/spec/modules/agenticAudienceAdapter_spec.js` | Example tests—**rename and align** with your module |
| `integration/submodules.json.snippet` | Illustrative `.submodules.json` entry pattern |
| `examples/publisher-config.example.js` | Illustrative only; real publisher setup lives with Prebid docs |

## Where usage is documented

- **Build, `setConfig`, and operational details:** [prebid/Prebid.js](https://github.com/prebid/Prebid.js) and [docs.prebid.org](https://docs.prebid.org/).
