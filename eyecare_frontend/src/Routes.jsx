import React from "react";
import { BrowserRouter, Routes as RouterRoutes, Route, Navigate } from "react-router-dom";
import ScrollToTop from "./components/ScrollToTop";
import ErrorBoundary from "./components/ErrorBoundary";
import ProtectedRoute from "./components/ProtectedRoute";
import NotFound from "./pages/NotFound";
import HomeDashboard from './pages/home-dashboard';
import PatientHistory from './pages/patient-history';
import ImageUploadAnalysis from './pages/image-upload-analysis';
import Login from './pages/login';
import Register from './pages/register';
import Profile from './pages/profile';
import Settings from './pages/settings';

const Routes = () => {
  return (
    <BrowserRouter>
      <ErrorBoundary>
      <ScrollToTop />
      <RouterRoutes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        
        {/* Protected routes */}
        <Route path="/" element={<Navigate to="/home-dashboard" replace />} />
        <Route 
          path="/home-dashboard" 
          element={
            <ProtectedRoute>
              <HomeDashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/patient-history" 
          element={
            <ProtectedRoute>
              <PatientHistory />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/image-upload-analysis" 
          element={
            <ProtectedRoute>
              <ImageUploadAnalysis />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/profile" 
          element={
            <ProtectedRoute>
              <Profile />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/settings" 
          element={
            <ProtectedRoute>
              <Settings />
            </ProtectedRoute>
          } 
        />
        
        <Route path="*" element={<NotFound />} />
      </RouterRoutes>
      </ErrorBoundary>
    </BrowserRouter>
  );
};

export default Routes;
