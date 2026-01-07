import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';
import Input from '../../../components/ui/Input';
import Select from '../../../components/ui/Select';

const HistoryFilters = ({ onFiltersChange, totalRecords }) => {
  const [filters, setFilters] = useState({
    dateFrom: '',
    dateTo: '',
    condition: '',
    confidenceMin: '',
    confidenceMax: '',
    searchQuery: ''
  });

  const conditionOptions = [
    { value: '', label: 'All Conditions' },
    { value: 'cataracts', label: 'Cataracts' },
    { value: 'glaucoma', label: 'Glaucoma' },
    { value: 'diabetic_retinopathy', label: 'Diabetic Retinopathy' },
    { value: 'healthy', label: 'Healthy' },
    { value: 'normal', label: 'Normal' },
    { value: 'macular_degeneration', label: 'Macular Degeneration' }
  ];

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    onFiltersChange(newFilters);
  };

  const clearFilters = () => {
    const clearedFilters = {
      dateFrom: '',
      dateTo: '',
      condition: '',
      confidenceMin: '',
      confidenceMax: '',
      searchQuery: ''
    };
    setFilters(clearedFilters);
    onFiltersChange(clearedFilters);
  };

  const hasActiveFilters = Object.values(filters)?.some(value => value !== '');

  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-6 shadow-card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Icon name="Filter" size={20} className="text-muted-foreground" />
          <h3 className="text-lg font-semibold text-foreground">Filter History</h3>
        </div>
        <div className="text-sm text-muted-foreground">
          {totalRecords} total records
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-4">
        {/* Search */}
        <div className="lg:col-span-2">
          <Input
            type="search"
            placeholder="Search by condition, notes, or ID..."
            value={filters?.searchQuery}
            onChange={(e) => handleFilterChange('searchQuery', e?.target?.value)}
            className="w-full"
          />
        </div>

        {/* Date From */}
        <Input
          type="date"
          label="From Date"
          value={filters?.dateFrom}
          onChange={(e) => handleFilterChange('dateFrom', e?.target?.value)}
        />

        {/* Date To */}
        <Input
          type="date"
          label="To Date"
          value={filters?.dateTo}
          onChange={(e) => handleFilterChange('dateTo', e?.target?.value)}
        />

        {/* Condition Filter */}
        <Select
          label="Condition"
          options={conditionOptions}
          value={filters?.condition}
          onChange={(value) => handleFilterChange('condition', value)}
        />

        {/* Confidence Min */}
        <Input
          type="number"
          label="Min Confidence %"
          placeholder="0"
          min="0"
          max="100"
          value={filters?.confidenceMin}
          onChange={(e) => handleFilterChange('confidenceMin', e?.target?.value)}
        />

        {/* Confidence Max */}
        <Input
          type="number"
          label="Max Confidence %"
          placeholder="100"
          min="0"
          max="100"
          value={filters?.confidenceMax}
          onChange={(e) => handleFilterChange('confidenceMax', e?.target?.value)}
        />

        {/* Clear Filters Button */}
        <div className="flex items-end">
          <Button
            variant="outline"
            onClick={clearFilters}
            disabled={!hasActiveFilters}
            iconName="X"
            iconPosition="left"
            className="w-full"
          >
            Clear Filters
          </Button>
        </div>
      </div>
      {hasActiveFilters && (
        <div className="flex flex-wrap gap-2 pt-4 border-t border-border">
          <span className="text-sm text-muted-foreground">Active filters:</span>
          {filters?.searchQuery && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-primary/10 text-primary">
              Search: {filters?.searchQuery}
            </span>
          )}
          {filters?.dateFrom && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-primary/10 text-primary">
              From: {filters?.dateFrom}
            </span>
          )}
          {filters?.dateTo && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-primary/10 text-primary">
              To: {filters?.dateTo}
            </span>
          )}
          {filters?.condition && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-primary/10 text-primary">
              Condition: {conditionOptions?.find(opt => opt?.value === filters?.condition)?.label}
            </span>
          )}
          {(filters?.confidenceMin || filters?.confidenceMax) && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-primary/10 text-primary">
              Confidence: {filters?.confidenceMin || '0'}% - {filters?.confidenceMax || '100'}%
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default HistoryFilters;