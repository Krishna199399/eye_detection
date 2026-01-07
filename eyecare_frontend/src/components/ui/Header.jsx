import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import Icon from '../AppIcon';
import Button from './Button';

const Header = () => {
  const [isMoreMenuOpen, setIsMoreMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const navigate = useNavigate();
  const { user, logout, isAuthenticated } = useAuth();

  const toggleMoreMenu = () => {
    setIsMoreMenuOpen(!isMoreMenuOpen);
  };

  const handleNavigation = (path) => {
    navigate(path);
    setIsMoreMenuOpen(false);
  };

  const handleLogout = async () => {
    try {
      await logout();
      setIsUserMenuOpen(false);
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // Get user initials for avatar
  const getUserInitials = () => {
    if (!user) return 'U';
    const firstName = user.first_name || user.full_name?.split(' ')[0] || '';
    const lastName = user.last_name || user.full_name?.split(' ')[1] || '';
    return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase() || 'U';
  };

  const getUserName = () => {
    if (!user) return 'User';
    if (user.first_name && user.last_name) {
      return `${user.first_name} ${user.last_name}`;
    }
    return user.full_name || user.email || 'User';
  };


  return (
    <header className="bg-card border-b border-border shadow-card sticky top-0 z-50">
      <div className="flex items-center justify-between h-16 px-6">
        {/* Logo Section */}
        <div className="flex items-center">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 flex items-center justify-center">
              <img 
                src="/assets/eyecenter-logo.png" 
                alt="EyeCenter Logo" 
                className="w-full h-full object-contain"
                onError={(e) => {
                  // Fallback to icon if image fails to load
                  e.target.style.display = 'none';
                  e.target.nextElementSibling.style.display = 'block';
                }}
              />
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center" style={{display: 'none'}}>
                <Icon name="Eye" size={20} color="white" />
              </div>
            </div>
            <div className="flex flex-col">
              <h1 className="text-lg font-heading font-semibold text-foreground">
                EyeCenter
              </h1>
              <span className="text-xs font-caption text-muted-foreground -mt-1">
                Your vision, our focus
              </span>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="hidden md:flex items-center space-x-1">
          <nav className="flex items-center space-x-1">
            <Button
              variant="ghost"
              onClick={() => handleNavigation('/home-dashboard')}
              className="text-sm font-body"
            >
              Dashboard
            </Button>
            <Button
              variant="ghost"
              onClick={() => handleNavigation('/image-upload-analysis')}
              className="text-sm font-body"
            >
              Analysis
            </Button>
            <Button
              variant="ghost"
              onClick={() => handleNavigation('/patient-history')}
              className="text-sm font-body"
            >
              History
            </Button>
            
            {/* More Menu */}
            <div className="relative">
              <Button
                variant="ghost"
                onClick={toggleMoreMenu}
                iconName="MoreHorizontal"
                iconSize={16}
                className="text-sm font-body"
              >
                More
              </Button>
              
              {isMoreMenuOpen && (
                <div className="absolute right-0 top-full mt-1 w-48 bg-popover border border-border rounded-lg shadow-elevated z-50">
                  <div className="py-1">
                    <button
                      onClick={() => handleNavigation('/settings')}
                      className="w-full px-4 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted transition-smooth flex items-center space-x-2"
                    >
                      <Icon name="Settings" size={16} />
                      <span>Settings</span>
                    </button>
                    <button
                      onClick={() => handleNavigation('/help')}
                      className="w-full px-4 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted transition-smooth flex items-center space-x-2"
                    >
                      <Icon name="HelpCircle" size={16} />
                      <span>Help & Support</span>
                    </button>
                    <div className="border-t border-border my-1"></div>
                    <button
                      onClick={() => handleNavigation('/about')}
                      className="w-full px-4 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted transition-smooth flex items-center space-x-2"
                    >
                      <Icon name="Info" size={16} />
                      <span>About</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </nav>

          {/* User Profile Menu */}
          {isAuthenticated && user && (
            <div className="relative ml-3">
              <button
                onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                className="flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-muted transition-colors"
              >
                <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white text-sm font-semibold">
                  {getUserInitials()}
                </div>
                <Icon name="ChevronDown" size={16} className="text-muted-foreground" />
              </button>

              {isUserMenuOpen && (
                <div className="absolute right-0 top-full mt-1 w-64 bg-popover border border-border rounded-lg shadow-elevated z-50">
                  {/* User Info */}
                  <div className="px-4 py-3 border-b border-border">
                    <p className="text-sm font-semibold text-foreground">{getUserName()}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{user.email}</p>
                    {user.user_type && (
                      <span className="inline-block mt-2 px-2 py-0.5 text-xs rounded-full bg-primary/10 text-primary capitalize">
                        {user.user_type.replace('_', ' ')}
                      </span>
                    )}
                  </div>

                  {/* Menu Items */}
                  <div className="py-1">
                    <button
                      onClick={() => {
                        handleNavigation('/profile');
                        setIsUserMenuOpen(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted transition-smooth flex items-center space-x-2"
                    >
                      <Icon name="User" size={16} />
                      <span>My Profile</span>
                    </button>
                    <button
                      onClick={() => {
                        handleNavigation('/settings');
                        setIsUserMenuOpen(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted transition-smooth flex items-center space-x-2"
                    >
                      <Icon name="Settings" size={16} />
                      <span>Settings</span>
                    </button>
                  </div>

                  {/* Logout */}
                  <div className="border-t border-border py-1">
                    <button
                      onClick={handleLogout}
                      className="w-full px-4 py-2 text-left text-sm font-body text-destructive hover:bg-destructive/10 transition-smooth flex items-center space-x-2"
                    >
                      <Icon name="LogOut" size={16} />
                      <span>Sign Out</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Mobile Menu Button */}
        <div className="md:hidden">
          <Button
            variant="ghost"
            onClick={toggleMoreMenu}
            iconName="Menu"
            iconSize={20}
          />
        </div>

        {/* Mobile Menu */}
        {isMoreMenuOpen && (
          <div className="md:hidden absolute top-16 left-0 right-0 bg-popover border-b border-border shadow-elevated z-50">
            <div className="px-4 py-2 space-y-1">
              <button
                onClick={() => handleNavigation('/home-dashboard')}
                className="w-full px-3 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted rounded-md transition-smooth flex items-center space-x-2"
              >
                <Icon name="LayoutDashboard" size={16} />
                <span>Dashboard</span>
              </button>
              <button
                onClick={() => handleNavigation('/image-upload-analysis')}
                className="w-full px-3 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted rounded-md transition-smooth flex items-center space-x-2"
              >
                <Icon name="Upload" size={16} />
                <span>Analysis</span>
              </button>
              <button
                onClick={() => handleNavigation('/patient-history')}
                className="w-full px-3 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted rounded-md transition-smooth flex items-center space-x-2"
              >
                <Icon name="History" size={16} />
                <span>History</span>
              </button>
              <div className="border-t border-border my-2"></div>
              <button
                onClick={() => handleNavigation('/settings')}
                className="w-full px-3 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted rounded-md transition-smooth flex items-center space-x-2"
              >
                <Icon name="Settings" size={16} />
                <span>Settings</span>
              </button>
              <button
                onClick={() => handleNavigation('/help')}
                className="w-full px-3 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted rounded-md transition-smooth flex items-center space-x-2"
              >
                <Icon name="HelpCircle" size={16} />
                <span>Help & Support</span>
              </button>
              <button
                onClick={() => handleNavigation('/about')}
                className="w-full px-3 py-2 text-left text-sm font-body text-popover-foreground hover:bg-muted rounded-md transition-smooth flex items-center space-x-2"
              >
                <Icon name="Info" size={16} />
                <span>About</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Overlay for mobile menu */}
      {isMoreMenuOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-20 z-40 md:hidden"
          onClick={() => setIsMoreMenuOpen(false)}
        />
      )}

      {/* Overlay for user menu (desktop) */}
      {isUserMenuOpen && (
        <div
          className="fixed inset-0 z-40 hidden md:block"
          onClick={() => setIsUserMenuOpen(false)}
        />
      )}
    </header>
  );
};

export default Header;