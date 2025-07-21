/**
 * Simple Client-Side Router for Parking Detection System
 * Handles navigation between different pages/views
 */

class Router {
    constructor(config = null) {
        this.routes = new Map();
        this.currentRoute = null;
        this.basePath = '';
        this.config = config;
        
        // Initialize router
        this.init();
    }

    /**
     * Initialize the router
     */
    init() {
        // Listen for browser back/forward button clicks
        window.addEventListener('popstate', (e) => {
            this.handleRouteChange();
        });

        // Listen for page load
        window.addEventListener('load', () => {
            this.handleRouteChange();
        });

        // Override default link behavior
        this.bindLinks();
    }

    /**
     * Register a route
     * @param {string} path - The route path
     * @param {Object} config - Route configuration
     */
    addRoute(path, config) {
        this.routes.set(path, {
            title: config.title || 'Page',
            action: config.action || (() => {}),
            beforeEnter: config.beforeEnter || null,
            afterEnter: config.afterEnter || null
        });
    }

    /**
     * Navigate to a specific route
     * @param {string} path - The route path to navigate to
     * @param {boolean} replace - Whether to replace current history entry
     */
    navigate(path, replace = false) {
        const route = this.routes.get(path);
        
        if (!route) {
            console.warn(`Route not found: ${path}`);
            this.navigate('/404', true);
            return;
        }

        // Execute beforeEnter hook if exists
        if (route.beforeEnter && !route.beforeEnter()) {
            return; // Navigation cancelled by beforeEnter hook
        }

        // Update browser history
        if (replace) {
            history.replaceState({ path }, route.title, path);
        } else {
            history.pushState({ path }, route.title, path);
        }

        // Update page title
        document.title = route.title;

        // Execute route action
        route.action();

        // Execute afterEnter hook if exists
        if (route.afterEnter) {
            route.afterEnter();
        }

        this.currentRoute = path;
    }

    /**
     * Go back in history
     */
    back() {
        history.back();
    }

    /**
     * Go forward in history
     */
    forward() {
        history.forward();
    }

    /**
     * Refresh current route
     */
    refresh() {
        if (this.currentRoute) {
            this.navigate(this.currentRoute, true);
        }
    }

    /**
     * Handle route changes (from popstate or direct navigation)
     */
    handleRouteChange() {
        const path = this.getCurrentPath();
        console.log('Handle route change for path:', path);
        
        // If we're directly accessing a file (like index.html, roi-config.html), don't handle routing
        const filename = path.split('/').pop();
        if (filename && filename.includes('.html')) {
            console.log('Direct HTML file access, skipping router');
            return;
        }
        
        const route = this.routes.get(path);

        if (route) {
            document.title = route.title;
            route.action();
            this.currentRoute = path;
        } else {
            console.log('Route not found, redirecting to home');
            // Handle 404 or redirect to default route
            window.location.href = 'index.html';
        }
    }

    /**
     * Get current path from URL
     * @returns {string} Current path
     */
    getCurrentPath() {
        const path = window.location.pathname;
        return path === '/' ? '/' : path.replace(/\/$/, ''); // Remove trailing slash
    }

    /**
     * Bind click events to navigation links
     */
    bindLinks() {
        document.addEventListener('click', (e) => {
            const link = e.target.closest('[data-route]');
            if (link) {
                e.preventDefault();
                e.stopPropagation();
                const path = link.getAttribute('data-route');
                
                console.log('Navigation triggered for path:', path);
                
                // Map routes to actual HTML files and navigate directly
                const routeMapping = {
                    '/': 'index.html',
                    '/roi-config': 'roi-config.html'
                };
                
                const targetFile = routeMapping[path];
                if (targetFile) {
                    console.log('Navigating to:', targetFile);
                    // Use relative path to stay in the same directory
                    window.location.href = targetFile;
                } else {
                    console.warn(`No file mapping found for route: ${path}`);
                    // Fallback to router navigation
                    this.navigate(path);
                }
            }
        });
    }

    /**
     * Get route parameters from URL (for future enhancement)
     * @param {string} pattern - Route pattern with parameters
     * @param {string} path - Current path
     * @returns {Object} Parameters object
     */
    getParams(pattern, path) {
        const patternParts = pattern.split('/');
        const pathParts = path.split('/');
        const params = {};

        patternParts.forEach((part, index) => {
            if (part.startsWith(':')) {
                const paramName = part.slice(1);
                params[paramName] = pathParts[index];
            }
        });

        return params;
    }

    /**
     * Check if current route matches a pattern
     * @param {string} pattern - Pattern to match against
     * @returns {boolean} Whether route matches
     */
    isActive(pattern) {
        return this.currentRoute === pattern;
    }
}

// Create global router instance
const router = new Router();

// Define application routes
const routes = {
    // Main dashboard/home page
    '/': {
        title: 'YOLO Object Detection System',
        action: () => {
            // Direct navigation to index.html
            window.location.href = 'index.html';
        }
    },

    // ROI Configuration page
    '/roi-config': {
        title: 'ROI Configuration - Parking Detection',
        action: () => {
            // Direct navigation to roi-config.html
            window.location.href = 'roi-config.html';
        },
        beforeEnter: () => {
            // Optional: Add validation before entering ROI config
            return true;
        }
    },

    // 404 page
    '/404': {
        title: '404 - Page Not Found',
        action: () => {
            console.log('404 page loaded');
            // Could show a 404 message or redirect to home
            setTimeout(() => router.navigate('/'), 3000);
        }
    }
};

// Register all routes
Object.keys(routes).forEach(path => {
    router.addRoute(path, routes[path]);
});

// Navigation helper functions
const navigation = {
    /**
     * Navigate to home page
     */
    goHome() {
        window.location.href = 'index.html';
    },

    /**
     * Navigate to ROI configuration
     */
    goToRoiConfig() {
        window.location.href = 'roi-config.html';
    },

    /**
     * Go back to previous page
     */
    goBack() {
        history.back();
    },

    /**
     * Navigate to a specific page
     * @param {string} path - Path to navigate to
     */
    goTo(path) {
        // Map routes to actual HTML files
        const routeMapping = {
            '/': 'index.html',
            '/roi-config': 'roi-config.html'
        };
        
        const targetFile = routeMapping[path];
        if (targetFile) {
            console.log('Navigation helper - navigating to:', targetFile);
            // Use relative path to stay in the same directory
            window.location.href = targetFile;
        } else {
            console.warn(`No file mapping found for route: ${path}`);
            // Try direct navigation as fallback
            window.location.href = path;
        }
    },

    /**
     * Check if we're on a specific route
     * @param {string} path - Path to check
     * @returns {boolean} Whether we're on the specified route
     */
    isCurrentRoute(path) {
        const routeMapping = {
            '/': 'index.html',
            '/roi-config': 'roi-config.html'
        };
        
        const currentFile = window.location.pathname.split('/').pop() || 'index.html';
        const targetFile = routeMapping[path];
        
        return currentFile === targetFile;
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { router, navigation };
} else {
    // Make available globally
    window.router = router;
    window.navigation = navigation;
}
