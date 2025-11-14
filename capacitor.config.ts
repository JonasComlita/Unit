// capacitor.config.ts
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.yourname.unitgame', // CHANGE THIS to your domain
  appName: 'Unit Strategy',
  webDir: 'build',
  bundledWebRuntime: false,
  
  server: {
    androidScheme: 'https',
    // Uncomment for development with local API:
    // url: 'http://10.0.2.2:3000', // Android emulator
    // cleartext: true
  },

  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: '#1a1a2e',
      showSpinner: false,
      androidSpinnerStyle: 'small',
      iosSpinnerStyle: 'small',
      splashFullScreen: true,
      splashImmersive: true,
    },
    
    StatusBar: {
      style: 'dark',
      backgroundColor: '#1a1a2e'
    },

    Keyboard: {
      resize: 'none',
      style: 'dark',
      resizeOnFullScreen: true
    },

    Haptics: {
      // Enable haptic feedback
    },

    ScreenOrientation: {
      // Lock to landscape for better gameplay
    },

    App: {
      // App state management
    }
  },

  ios: {
    contentInset: 'always',
    preferredContentMode: 'mobile',
    scheme: 'App',
    // Minimum iOS version
    minVersion: '13.0'
  },

  android: {
    backgroundColor: '#1a1a2e',
    allowMixedContent: false,
    captureInput: true,
    webContentsDebuggingEnabled: false, // Set to true for debugging
    
    buildOptions: {
      keystorePath: undefined, // Set during release build
      keystorePassword: undefined,
      keystoreAlias: undefined,
      keystoreAliasPassword: undefined,
      releaseType: 'AAB' // Android App Bundle (recommended)
    }
  }
};

export default config;