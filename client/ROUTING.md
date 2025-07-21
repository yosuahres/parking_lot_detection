# JavaScript Routing System Documentation

## Overview

This routing system provides client-side navigation for the Parking Detection System interface. It handles navigation between different pages while maintaining browser history and providing a clean, programmatic way to manage routes.

## Files

- **`router.js`** - Core routing engine
- **`route-config.js`** - Route definitions and configuration
- **`routing-demo.html`** - Interactive demonstration of routing features
- **`index.html`** - Main dashboard (updated to use routing)
- **`roi-config.html`** - ROI configuration page (updated to use routing)

## Quick Start

### 1. Include Required Scripts

```html
<script src="route-config.js"></script>
<script src="router.js"></script>
```

### 2. Basic Navigation

```javascript
// Navigate to different pages
navigation.goHome();              // Go to dashboard
navigation.goToRoiConfig();       // Go to ROI configuration
navigation.goBack();              // Browser back button
navigation.goTo('/custom-path');  // Navigate to custom route
```

### 3. Check Current Route

```javascript
if (navigation.isCurrentRoute('/roi-config')) {
    console.log('Currently on ROI config page');
}
```

## Core Components

### Router Class

The main routing engine that handles:
- Route registration and navigation
- Browser history management
- Route lifecycle hooks (beforeEnter, afterEnter)
- Automatic link binding with `data-route` attributes

#### Methods

```javascript
// Add a new route
router.addRoute('/path', {
    title: 'Page Title',
    action: () => { /* route handler */ },
    beforeEnter: () => { /* validation */ return true; },
    afterEnter: () => { /* cleanup */ }
});

// Navigate to a route
router.navigate('/path', false); // Add to history
router.navigate('/path', true);  // Replace current history entry

// Browser navigation
router.back();     // Go back
router.forward();  // Go forward
router.refresh();  // Refresh current route
```

### Navigation Helper

Simplified navigation interface:

```javascript
navigation.goHome();           // Navigate to dashboard
navigation.goToRoiConfig();    // Navigate to ROI config
navigation.goBack();           // Go back
navigation.goTo('/path');      // Navigate to specific path
navigation.isCurrentRoute('/path'); // Check current route
```

### Route Configuration

Centralized route definitions in `route-config.js`:

```javascript
const RouteConfig = {
    routes: {
        '/': {
            name: 'home',
            title: 'Dashboard',
            file: 'index.html',
            description: 'Main dashboard',
            permissions: []
        },
        '/roi-config': {
            name: 'roi-config', 
            title: 'ROI Configuration',
            file: 'roi-config.html',
            description: 'Configure parking spots',
            permissions: []
        }
    }
};
```

### Route Utilities

Helper functions for route management:

```javascript
// Get route information
const route = RouteUtils.getRoute('/roi-config');
console.log(route.title); // "ROI Configuration"

// Check if route exists
if (RouteUtils.routeExists('/my-route')) {
    // Route exists
}

// Get all available routes
const allRoutes = RouteUtils.getAllRoutes();

// Build full URL for a route
const url = RouteUtils.buildUrl('/roi-config');

// Check permissions (future feature)
const canAccess = RouteUtils.canAccessRoute('/admin', userPermissions);
```

## HTML Integration

### Method 1: Using Data Attributes (Recommended)

```html
<button data-route="/roi-config">Go to ROI Config</button>
<a data-route="/">Back to Dashboard</a>
```

### Method 2: Using Navigation Functions

```html
<button onclick="navigation.goToRoiConfig()">ROI Config</button>
<button onclick="navigation.goBack()">Back</button>
```

### Method 3: Using Router Directly

```html
<button onclick="router.navigate('/roi-config')">ROI Config</button>
```

## Advanced Features

### Route Lifecycle Hooks

```javascript
router.addRoute('/protected', {
    title: 'Protected Page',
    beforeEnter: () => {
        // Validation before entering route
        if (!user.isAuthenticated()) {
            router.navigate('/login');
            return false; // Cancel navigation
        }
        return true;
    },
    action: () => {
        // Main route handler
        loadProtectedContent();
    },
    afterEnter: () => {
        // Cleanup after route enters
        trackPageView('/protected');
    }
});
```

### Middleware Integration

```javascript
// Check backend health before navigation
RouteConfig.middleware.checkBackendHealth().then(isHealthy => {
    if (!isHealthy) {
        showError('Backend is not available');
    }
});

// Log navigation events
RouteConfig.middleware.logNavigation('/from', '/to');
```

### Error Handling

```javascript
// 404 handling is built-in
// When route not found, redirects to default route

// Custom error handling
router.addRoute('/404', {
    title: '404 - Not Found',
    action: () => {
        showErrorPage('Page not found');
        setTimeout(() => router.navigate('/'), 3000);
    }
});
```

## Current Implementation

### Dashboard (index.html)
- Route: `/`
- Features: Main detection interface, system status, controls
- Navigation: "Setup Parking Areas" button → `/roi-config`

### ROI Configuration (roi-config.html)  
- Route: `/roi-config`
- Features: Video/image loading, parking spot marking, JSON export
- Navigation: "Back to Main" button → `/` (home)

## Browser Compatibility

- Modern browsers with HTML5 History API support
- Graceful fallback for older browsers
- Mobile responsive navigation

## Development

### Adding New Routes

1. **Define route in `route-config.js`:**
```javascript
'/new-page': {
    name: 'new-page',
    title: 'New Page',
    file: 'new-page.html',
    description: 'Description of new page'
}
```

2. **Add navigation helper (optional):**
```javascript
navigation.goToNewPage = () => router.navigate('/new-page');
```

3. **Create HTML file with routing scripts:**
```html
<script src="route-config.js"></script>
<script src="router.js"></script>
```

4. **Initialize in page script:**
```javascript
window.addEventListener('load', () => {
    router.currentRoute = '/new-page';
});
```

### Testing

Open `routing-demo.html` in your browser to:
- Test navigation methods
- View route information
- See configuration details
- Try different navigation patterns

## Future Enhancements

- [ ] URL parameter parsing (`/route/:id`)
- [ ] Query string support
- [ ] Route guards and permissions
- [ ] Animated page transitions
- [ ] Deep linking support
- [ ] Route prefetching
- [ ] Breadcrumb generation

## Troubleshooting

### Common Issues

1. **Routes not working:**
   - Ensure both `route-config.js` and `router.js` are loaded
   - Check browser console for JavaScript errors
   - Verify route is defined in RouteConfig

2. **Navigation not updating:**
   - Make sure `router.currentRoute` is set in page load handlers
   - Check for conflicting event handlers

3. **Back button not working:**
   - Router automatically handles popstate events
   - Ensure no other code is interfering with history

### Debug Mode

```javascript
// Enable debug logging
router.debug = true;

// Check current state
console.log('Current route:', router.currentRoute);
console.log('Available routes:', Array.from(router.routes.keys()));
```

## License

Part of the Parking Detection System project.
