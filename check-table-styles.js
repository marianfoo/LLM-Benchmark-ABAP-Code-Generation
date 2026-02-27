#!/usr/bin/env node

const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  
  // Set viewport to 390px wide (mobile)
  await page.setViewport({ width: 390, height: 844 });
  
  // Navigate to localhost:8000
  await page.goto('http://localhost:8000/', { waitUntil: 'networkidle0' });
  
  // Wait for the table to be rendered
  await page.waitForSelector('#mainTable');
  
  // Execute the diagnostic code
  const diagnostics = await page.evaluate(() => {
    const t = document.querySelector('#mainTable');
    const style = window.getComputedStyle(t);
    
    const th = t.querySelector('th');
    const thStyle = th ? window.getComputedStyle(th) : null;
    
    const wrap = t.closest('.table-wrap');
    const wrapStyle = wrap ? window.getComputedStyle(wrap) : null;
    
    return {
      table: {
        width: style.width,
        minWidth: style.minWidth,
        display: style.display,
      },
      th: thStyle ? {
        whiteSpace: thStyle.whiteSpace,
        width: thStyle.width,
        writingMode: thStyle.writingMode,
        transform: thStyle.transform,
      } : null,
      wrap: wrapStyle ? {
        overflowX: wrapStyle.overflowX,
        width: wrapStyle.width,
        display: wrapStyle.display,
      } : null,
    };
  });
  
  console.log('=== DIAGNOSTIC RESULTS ===\n');
  console.log('TABLE (#mainTable):');
  console.log('  width:', diagnostics.table.width);
  console.log('  min-width:', diagnostics.table.minWidth);
  console.log('  display:', diagnostics.table.display);
  console.log('');
  
  if (diagnostics.th) {
    console.log('TH (first <th>):');
    console.log('  white-space:', diagnostics.th.whiteSpace);
    console.log('  width:', diagnostics.th.width);
    console.log('  writing-mode:', diagnostics.th.writingMode);
    console.log('  transform:', diagnostics.th.transform);
    console.log('');
  }
  
  if (diagnostics.wrap) {
    console.log('WRAP (.table-wrap):');
    console.log('  overflow-x:', diagnostics.wrap.overflowX);
    console.log('  width:', diagnostics.wrap.width);
    console.log('  display:', diagnostics.wrap.display);
  }
  
  await browser.close();
})();
