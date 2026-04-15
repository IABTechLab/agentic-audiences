/**
 * Example Prebid.js publisher configuration for the Agentic Audiences RTD submodule.
 * Copy into your site’s Prebid bootstrap. Replace the GVL placeholder with your
 * vendor’s numeric Global Vendor List ID (see README.md in this folder).
 */

pbjs.setConfig({
  consentManagement: {
    // ... your CMP / GDPR config as required
  },
  gvlMapping: {
    // Key must match the RTD data provider name: "agenticAudience"
    agenticAudience: 0, // TODO: replace 0 with your vendor’s numeric GVL ID (not a string)
  },
  realTimeData: {
    dataProviders: [
      {
        name: 'agenticAudience',
        params: {
          // Optional: use a dedicated storage key for your integration
          // storageKey: '_my_vendor_agentic_audience_',
        },
      },
    ],
  },
});
