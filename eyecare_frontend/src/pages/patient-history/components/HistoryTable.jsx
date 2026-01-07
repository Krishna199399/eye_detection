import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';

const HistoryTable = ({ historyData, onViewDetails, onDownloadReport, onCompareAnalysis }) => {
  const [selectedItems, setSelectedItems] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: 'date', direction: 'desc' });

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig?.key === key && sortConfig?.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const handleSelectItem = (id) => {
    setSelectedItems(prev => 
      prev?.includes(id) 
        ? prev?.filter(item => item !== id)
        : [...prev, id]
    );
  };

  const handleSelectAll = () => {
    if (selectedItems?.length === historyData?.length) {
      setSelectedItems([]);
    } else {
      setSelectedItems(historyData?.map(item => item?.id));
    }
  };

  const getConditionBadge = (condition, confidence) => {
    const conditionStyles = {
      'healthy': 'bg-success/10 text-success border-success/20',
      'cataracts': 'bg-warning/10 text-warning border-warning/20',
      'glaucoma': 'bg-destructive/10 text-destructive border-destructive/20',
      'diabetic_retinopathy': 'bg-accent/10 text-accent border-accent/20',
      'macular_degeneration': 'bg-secondary/10 text-secondary border-secondary/20'
    };

    return (
      <div className="flex flex-col space-y-1">
        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${conditionStyles?.[condition] || 'bg-muted/10 text-muted-foreground border-muted/20'}`}>
          {condition?.replace('_', ' ')?.replace(/\b\w/g, l => l?.toUpperCase())}
        </span>
        <span className="text-xs text-muted-foreground">{confidence}% confidence</span>
      </div>
    );
  };

  const sortedData = [...historyData]?.sort((a, b) => {
    if (sortConfig?.key === 'date') {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return sortConfig?.direction === 'asc' ? dateA - dateB : dateB - dateA;
    }
    if (sortConfig?.key === 'confidence') {
      return sortConfig?.direction === 'asc' ? a?.confidence - b?.confidence : b?.confidence - a?.confidence;
    }
    if (sortConfig?.key === 'condition') {
      return sortConfig?.direction === 'asc' 
        ? a?.condition?.localeCompare(b?.condition)
        : b?.condition?.localeCompare(a?.condition);
    }
    return 0;
  });

  const getSortIcon = (key) => {
    if (sortConfig?.key !== key) return 'ArrowUpDown';
    return sortConfig?.direction === 'asc' ? 'ArrowUp' : 'ArrowDown';
  };

  return (
    <div className="bg-card border border-border rounded-lg shadow-card overflow-hidden">
      {/* Table Header with Bulk Actions */}
      {selectedItems?.length > 0 && (
        <div className="bg-primary/5 border-b border-border px-6 py-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">
              {selectedItems?.length} item{selectedItems?.length > 1 ? 's' : ''} selected
            </span>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                iconName="Download"
                iconPosition="left"
                onClick={() => console.log('Bulk download', selectedItems)}
              >
                Download Reports
              </Button>
              <Button
                variant="outline"
                size="sm"
                iconName="BarChart3"
                iconPosition="left"
                onClick={() => console.log('Compare selected', selectedItems)}
              >
                Compare
              </Button>
            </div>
          </div>
        </div>
      )}
      {/* Desktop Table View */}
      <div className="hidden lg:block overflow-x-auto">
        <table className="w-full">
          <thead className="bg-muted/30 border-b border-border">
            <tr>
              <th className="w-12 px-6 py-4">
                <input
                  type="checkbox"
                  checked={selectedItems?.length === historyData?.length}
                  onChange={handleSelectAll}
                  className="rounded border-border"
                />
              </th>
              <th className="px-6 py-4 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Image
              </th>
              <th className="px-6 py-4 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                <button
                  onClick={() => handleSort('date')}
                  className="flex items-center space-x-1 hover:text-foreground transition-colors"
                >
                  <span>Date</span>
                  <Icon name={getSortIcon('date')} size={14} />
                </button>
              </th>
              <th className="px-6 py-4 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                <button
                  onClick={() => handleSort('condition')}
                  className="flex items-center space-x-1 hover:text-foreground transition-colors"
                >
                  <span>Condition</span>
                  <Icon name={getSortIcon('condition')} size={14} />
                </button>
              </th>
              <th className="px-6 py-4 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                <button
                  onClick={() => handleSort('confidence')}
                  className="flex items-center space-x-1 hover:text-foreground transition-colors"
                >
                  <span>Confidence</span>
                  <Icon name={getSortIcon('confidence')} size={14} />
                </button>
              </th>
              <th className="px-6 py-4 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Notes
              </th>
              <th className="px-6 py-4 text-right text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {sortedData?.map((item) => (
              <tr key={item?.id} className="hover:bg-muted/20 transition-colors">
                <td className="px-6 py-4">
                  <input
                    type="checkbox"
                    checked={selectedItems?.includes(item?.id)}
                    onChange={() => handleSelectItem(item?.id)}
                    className="rounded border-border"
                  />
                </td>
                <td className="px-6 py-4">
                  <div className="w-16 h-16 rounded-lg overflow-hidden bg-muted">
                    <Image
                      src={item?.imageUrl}
                      alt={`Eye scan ${item?.id}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-foreground">
                      {new Date(item.date)?.toLocaleDateString()}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(item.date)?.toLocaleTimeString()}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  {getConditionBadge(item?.condition, item?.confidence)}
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${item?.confidence >= 90 ? 'bg-success' : item?.confidence >= 70 ? 'bg-warning' : 'bg-destructive'}`}></div>
                    <span className="text-sm font-medium text-foreground">{item?.confidence}%</span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="text-sm text-muted-foreground max-w-xs truncate block">
                    {item?.notes || 'No notes'}
                  </span>
                </td>
                <td className="px-6 py-4 text-right">
                  <div className="flex items-center justify-end space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="Eye"
                      onClick={() => onViewDetails(item)}
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="Download"
                      onClick={() => onDownloadReport(item)}
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="BarChart3"
                      onClick={() => onCompareAnalysis(item)}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Mobile Card View */}
      <div className="lg:hidden divide-y divide-border">
        {sortedData?.map((item) => (
          <div key={item?.id} className="p-4">
            <div className="flex items-start space-x-4">
              <input
                type="checkbox"
                checked={selectedItems?.includes(item?.id)}
                onChange={() => handleSelectItem(item?.id)}
                className="mt-2 rounded border-border"
              />
              <div className="w-16 h-16 rounded-lg overflow-hidden bg-muted flex-shrink-0">
                <Image
                  src={item?.imageUrl}
                  alt={`Eye scan ${item?.id}`}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-sm font-medium text-foreground">
                      {new Date(item.date)?.toLocaleDateString()}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(item.date)?.toLocaleTimeString()}
                    </p>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="Eye"
                      onClick={() => onViewDetails(item)}
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="Download"
                      onClick={() => onDownloadReport(item)}
                    />
                  </div>
                </div>
                <div className="mt-2">
                  {getConditionBadge(item?.condition, item?.confidence)}
                </div>
                {item?.notes && (
                  <p className="mt-2 text-sm text-muted-foreground">
                    {item?.notes}
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
      {historyData?.length === 0 && (
        <div className="text-center py-12">
          <Icon name="FileX" size={48} className="mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">No history found</h3>
          <p className="text-muted-foreground">No diagnostic history matches your current filters.</p>
        </div>
      )}
    </div>
  );
};

export default HistoryTable;