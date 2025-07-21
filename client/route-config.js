/**
 * Route Configuration for Parking Detection System
 * Define all application routes and their behaviors
 */

const RouteConfig = {
    // Base URL for the application
    baseUrl: window.location.origin,
    
    // Base path for the application (adjust based on your deployment)
    basePath: window.location.pathname.replace(/\/[^\/]*$/, ''), // Gets the directory path
    
    // Default route when no route is specified
    defaultRoute: '/',
    
    // Route definitions
    routes: {
        '/': {
            name: 'home',
            title: 'YOLO Object Detection System',
            file: 'index.html',
            description: 'Main dashboard for parking detection system',
            permissions: [] // Add permission requirements if needed
        },
        
        '/roi-config': {
            name: 'roi-config',
            title: 'ROI Configuration - Parking Detection',
            file: 'roi-config.html',
            description: 'Configure regions of interest for parking spots',
            permissions: [] // Add permission requirements if needed
        }
    },
    
    // Navigation menu items (for future use)
    navigation: [
        {
            name: 'Dashboard',
            route: '/',
            icon: 'dashboard',
            visible: true
        },
        {
            name: 'ROI Setup',
            route: '/roi-config',
            icon: 'settings',
            visible: true
        }
    ],
    
    // Middleware functions
    middleware: {
        /**
         * Authentication check (placeholder for future implementation)
         * @returns {boolean} Whether user is authenticated
         */
        requireAuth: () => {
            // Placeholder for authentication logic
            return true;
        },
        
        /**
         * Check if backend is available
         * @returns {Promise<boolean>} Whether backend is reachable
         */
        checkBackendHealth: async () => {
            try {
                const response = await fetch('http://localhost:8000/status', { 
                    method: 'GET',
                    timeout: 5000
                });
                return response.ok;
            } catch (error) {
                console.warn('Backend health check failed:', error);
                return false;
            }
        },
        
        /**
         * Log navigation events
         * @param {string} from - Previous route
         * @param {string} to - New route
         */
        logNavigation: (from, to) => {
            console.log(`Navigation: ${from} → ${to}`);
        }
    },
    
    // Error handling configuration
    errorHandling: {
        404: {
            title: '404 - Page Not Found',
            message: 'The requested page could not be found.',
            redirectAfter: 3000, // Redirect after 3 seconds
            redirectTo: '/'
        },
        
        500: {
            title: '500 - Server Error',
            message: 'An internal server error occurred.',
            redirectAfter: 5000,
            redirectTo: '/'
        }
    },
    
    // Animation and transition settings
    transitions: {
        enabled: true,
        duration: 300, // milliseconds
        easing: 'ease-in-out'
    }
};

// Utility functions for route management
const RouteUtils = {
    /**
     * Get route configuration by path
     * @param {string} path - Route path
     * @returns {Object|null} Route configuration or null if not found
     */
    getRoute(path) {
        return RouteConfig.routes[path] || null;
    },
    
    /**
     * Get all available routes
     * @returns {Array} Array of route objects with path and config
     */
    getAllRoutes() {
        return Object.keys(RouteConfig.routes).map(path => ({
            path,
            ...RouteConfig.routes[path]
        }));
    },
    
    /**
     * Check if a route exists
     * @param {string} path - Route path
     * @returns {boolean} Whether route exists
     */
    routeExists(path) {
        return path in RouteConfig.routes;
    },
    
    /**
     * Get navigation menu items
     * @param {boolean} visibleOnly - Return only visible items
     * @returns {Array} Navigation items
     */
    getNavigationItems(visibleOnly = true) {
        return visibleOnly 
            ? RouteConfig.navigation.filter(item => item.visible)
            : RouteConfig.navigation;
    },
    
    /**
     * Build full URL for a route
     * @param {string} path - Route path
     * @returns {string} Full URL
     */
    buildUrl(path) {
        const route = this.getRoute(path);
        if (!route) return null;
        
        // Use the current directory path as base
        const currentPath = window.location.pathname;
        const basePath = currentPath.substring(0, currentPath.lastIndexOf('/') + 1);
        
        return `${RouteConfig.baseUrl}${basePath}${route.file}`;
    },
    
    /**
     * Validate route permissions (placeholder for future implementation)
     * @param {string} path - Route path
     * @param {Object} userPermissions - User's permissions
     * @returns {boolean} Whether user can access route
     */
    canAccessRoute(path, userPermissions = []) {
        const route = this.getRoute(path);
        if (!route) return false;
        
        // If no permissions required, allow access
        if (!route.permissions || route.permissions.length === 0) {
            return true;
        }
        
        // Check if user has required permissions
        return route.permissions.every(permission => 
            userPermissions.includes(permission)
        );
    }
};

// Export configurations
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RouteConfig, RouteUtils };
} else {
    // Make available globally
    window.RouteConfig = RouteConfig;
    window.RouteUtils = RouteUtils;
}
