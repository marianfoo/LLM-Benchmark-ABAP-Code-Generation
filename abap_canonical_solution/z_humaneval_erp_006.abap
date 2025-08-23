CLASS z_humaneval_erp_006 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: calculate_stock
      IMPORTING
        mat_id        TYPE char20
      RETURNING
        VALUE(rs_result) TYPE ztransaction.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_erp_006 IMPLEMENTATION.

  METHOD calculate_stock.
    DATA: lv_current_stock TYPE i,
          lt_transactions  TYPE TABLE OF ztransaction,
          ls_transaction   TYPE ztransaction,
          lv_shortage      TYPE i.

    " Initialize result structure
    CLEAR rs_result.

    " Get initial stock for the material from zstock table
    SELECT SINGLE stock
      FROM zstock
      INTO lv_current_stock
      WHERE mat_id = mat_id.

    IF sy-subrc <> 0.
      " Material not found in stock table - assume 0 stock
      lv_current_stock = 0.
    ENDIF.

    " Get all transactions for the given material, sorted by date
    SELECT *
      FROM ztransaction
      INTO TABLE lt_transactions
      WHERE mat_id = mat_id
      ORDER BY transaction_date.

    " Process each transaction in chronological order
    LOOP AT lt_transactions INTO ls_transaction.
      " Adjust stock based on transaction amount
      lv_current_stock = lv_current_stock + ls_transaction-amount.

      " Check if stock drops below zero
      IF lv_current_stock < 0.
        " Calculate shortage amount (positive number)
        lv_shortage = abs( lv_current_stock ).

        " Create new planning order to cover the shortage
        rs_result-id = 999999.
        rs_result-mat_id = mat_id.
        rs_result-transaction_type = 'PLO'.
        rs_result-transaction_date = ls_transaction-transaction_date.
        rs_result-amount = lv_shortage.

        " Insert the new PLO into ztransaction table
        INSERT ztransaction FROM rs_result.
        COMMIT WORK.

        " Adjust current stock to zero (shortage is covered by PLO)
        lv_current_stock = 0.

        " Exit loop as we found the first shortage
        EXIT.
      ENDIF.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.