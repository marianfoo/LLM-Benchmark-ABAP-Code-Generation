CLASS ltcl_stock_calculator_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      setup,
      teardown,
      test_no_shortage FOR TESTING,
      test_shortage_occurs FOR TESTING,
      test_material_not_found FOR TESTING,
      test_no_transactions FOR TESTING,
      test_multiple_transactions FOR TESTING,
      test_zero_initial_stock FOR TESTING.

    DATA: lt_stock_backup TYPE TABLE OF zstock,
          lt_trans_backup TYPE TABLE OF ztransaction.

ENDCLASS.

CLASS ltcl_stock_calculator_test IMPLEMENTATION.

  METHOD setup.
    " Backup existing data
    SELECT * FROM zstock INTO TABLE lt_stock_backup.
    SELECT * FROM ztransaction INTO TABLE lt_trans_backup.

    " Clear tables for clean test environment
    DELETE FROM zstock.
    DELETE FROM ztransaction.
    COMMIT WORK.
  ENDMETHOD.

  METHOD teardown.
    " Restore original data
    DELETE FROM zstock.
    DELETE FROM ztransaction.

    IF lines( lt_stock_backup ) > 0.
      INSERT zstock FROM TABLE lt_stock_backup.
    ENDIF.

    IF lines( lt_trans_backup ) > 0.
      INSERT ztransaction FROM TABLE lt_trans_backup.
    ENDIF.

    COMMIT WORK.
  ENDMETHOD.

  METHOD test_no_shortage.
    " Test case: sufficient stock, no shortage occurs
    DATA: ls_stock  TYPE zstock,
          ls_trans  TYPE ztransaction,
          ls_result TYPE ztransaction.

    " Setup test data
    ls_stock-mat_id = 'MAT001'.
    ls_stock-stock = 10.
    INSERT zstock FROM ls_stock.

    ls_trans-id = 1.
    ls_trans-mat_id = 'MAT001'.
    ls_trans-transaction_type = 'RES'.
    ls_trans-transaction_date = '20240620'.
    ls_trans-amount = -5.
    INSERT ztransaction FROM ls_trans.

    ls_trans-id = 2.
    ls_trans-transaction_type = 'PLO'.
    ls_trans-transaction_date = '20240625'.
    ls_trans-amount = 3.
    INSERT ztransaction FROM ls_trans.

    COMMIT WORK.

    " Execute method
    ls_result = z_humaneval_erp_006=>calculate_stock( mat_id = 'MAT001' ).

    " Assert: no PLO should be created (result should be initial)
    cl_abap_unit_assert=>assert_initial(
      act = ls_result-id
      msg = 'No PLO should be created when no shortage occurs'
    ).
  ENDMETHOD.

  METHOD test_shortage_occurs.
    " Test case: shortage occurs, PLO should be created
    DATA: ls_stock  TYPE zstock,
          ls_trans  TYPE ztransaction,
          ls_result TYPE ztransaction.

    " Setup test data - same as example in requirements
    ls_stock-mat_id = 'X'.
    ls_stock-stock = 3.
    INSERT zstock FROM ls_stock.

    ls_trans-id = 1.
    ls_trans-mat_id = 'X'.
    ls_trans-transaction_type = 'RES'.
    ls_trans-transaction_date = '20240620'.
    ls_trans-amount = -2.
    INSERT ztransaction FROM ls_trans.

    ls_trans-id = 2.
    ls_trans-transaction_type = 'PLO'.
    ls_trans-transaction_date = '20240625'.
    ls_trans-amount = 1.
    INSERT ztransaction FROM ls_trans.

    ls_trans-id = 3.
    ls_trans-transaction_type = 'RES'.
    ls_trans-transaction_date = '20240630'.
    ls_trans-amount = -3.
    INSERT ztransaction FROM ls_trans.

    COMMIT WORK.

    " Execute method
    ls_result = z_humaneval_erp_006=>calculate_stock( mat_id = 'X' ).

    " Assert: PLO should be created
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-id
      exp = 999999
      msg = 'PLO ID should be 999999'
    ).

    cl_abap_unit_assert=>assert_equals(
      act = ls_result-mat_id
      exp = 'X'
      msg = 'Material ID should match'
    ).

    cl_abap_unit_assert=>assert_equals(
      act = ls_result-transaction_type
      exp = 'PLO'
      msg = 'Transaction type should be PLO'
    ).

    cl_abap_unit_assert=>assert_equals(
      act = ls_result-transaction_date
      exp = '20240630'
      msg = 'Date should be when shortage occurs'
    ).

    cl_abap_unit_assert=>assert_equals(
      act = ls_result-amount
      exp = 1
      msg = 'Amount should be shortage quantity (positive)'
    ).
  ENDMETHOD.

  METHOD test_material_not_found.
    " Test case: material not in stock table
    DATA: ls_trans  TYPE ztransaction,
          ls_result TYPE ztransaction.

    " Setup transaction for non-existent material
    ls_trans-id = 1.
    ls_trans-mat_id = 'NOTFOUND'.
    ls_trans-transaction_type = 'RES'.
    ls_trans-transaction_date = '20240620'.
    ls_trans-amount = -1.
    INSERT ztransaction FROM ls_trans.

    COMMIT WORK.

    " Execute method
    ls_result = z_humaneval_erp_006=>calculate_stock( mat_id = 'NOTFOUND' ).

    " Assert: PLO should be created due to shortage (0 initial stock - 1 = -1)
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-id
      exp = 999999
      msg = 'PLO should be created for non-existent material'
    ).

    cl_abap_unit_assert=>assert_equals(
      act = ls_result-amount
      exp = 1
      msg = 'Shortage amount should be 1'
    ).
  ENDMETHOD.

  METHOD test_no_transactions.
    " Test case: material exists in stock but no transactions
    DATA: ls_stock  TYPE zstock,
          ls_result TYPE ztransaction.

    " Setup stock only
    ls_stock-mat_id = 'MAT002'.
    ls_stock-stock = 5.
    INSERT zstock FROM ls_stock.

    COMMIT WORK.

    " Execute method
    ls_result = z_humaneval_erp_006=>calculate_stock( mat_id = 'MAT002' ).

    " Assert: no PLO should be created
    cl_abap_unit_assert=>assert_initial(
      act = ls_result-id
      msg = 'No PLO should be created when no transactions exist'
    ).
  ENDMETHOD.

  METHOD test_multiple_transactions.
    " Test case: multiple transactions, shortage occurs on second transaction
    DATA: ls_stock  TYPE zstock,
          ls_trans  TYPE ztransaction,
          ls_result TYPE ztransaction.

    " Setup test data
    ls_stock-mat_id = 'MAT003'.
    ls_stock-stock = 5.
    INSERT zstock FROM ls_stock.

    " First transaction - no shortage
    ls_trans-id = 1.
    ls_trans-mat_id = 'MAT003'.
    ls_trans-transaction_type = 'RES'.
    ls_trans-transaction_date = '20240610'.
    ls_trans-amount = -2.
    INSERT ztransaction FROM ls_trans.

    " Second transaction - causes shortage
    ls_trans-id = 2.
    ls_trans-transaction_type = 'RES'.
    ls_trans-transaction_date = '20240615'.
    ls_trans-amount = -5.
    INSERT ztransaction FROM ls_trans.

    " Third transaction - should not be processed due to earlier shortage
    ls_trans-id = 3.
    ls_trans-transaction_type = 'PLO'.
    ls_trans-transaction_date = '20240620'.
    ls_trans-amount = 10.
    INSERT ztransaction FROM ls_trans.

    COMMIT WORK.

    " Execute method
    ls_result = z_humaneval_erp_006=>calculate_stock( mat_id = 'MAT003' ).

    " Assert: PLO should be created for shortage on second transaction
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-transaction_date
      exp = '20240615'
      msg = 'Date should be when first shortage occurs'
    ).

    cl_abap_unit_assert=>assert_equals(
      act = ls_result-amount
      exp = 2
      msg = 'Shortage amount should be 2 (5-2-5 = -2)'
    ).
  ENDMETHOD.

  METHOD test_zero_initial_stock.
    " Test case: zero initial stock with positive transaction
    DATA: ls_stock  TYPE zstock,
          ls_trans  TYPE ztransaction,
          ls_result TYPE ztransaction.

    " Setup zero stock
    ls_stock-mat_id = 'MAT004'.
    ls_stock-stock = 0.
    INSERT zstock FROM ls_stock.

    " Positive transaction
    ls_trans-id = 1.
    ls_trans-mat_id = 'MAT004'.
    ls_trans-transaction_type = 'PLO'.
    ls_trans-transaction_date = '20240620'.
    ls_trans-amount = 5.
    INSERT ztransaction FROM ls_trans.

    " Negative transaction causing shortage
    ls_trans-id = 2.
    ls_trans-transaction_type = 'RES'.
    ls_trans-transaction_date = '20240625'.
    ls_trans-amount = -8.
    INSERT ztransaction FROM ls_trans.

    COMMIT WORK.

    " Execute method
    ls_result = z_humaneval_erp_006=>calculate_stock( mat_id = 'MAT004' ).

    " Assert: PLO should be created
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-amount
      exp = 3
      msg = 'Shortage amount should be 3 (0+5-8 = -3)'
    ).
  ENDMETHOD.

ENDCLASS.