CLASS z_humaneval_142 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: sum_squares
      IMPORTING
        lst           TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_142 IMPLEMENTATION.

  METHOD sum_squares.
    DATA: lv_index TYPE i,
          lv_value TYPE i,
          lv_sum   TYPE i.

    LOOP AT lst INTO lv_value.
      lv_index = sy-tabix - 1.

      IF lv_index MOD 3 = 0.
        lv_sum = lv_sum + ( lv_value * lv_value ).
      ELSEIF lv_index MOD 4 = 0 AND lv_index MOD 3 <> 0.
        lv_sum = lv_sum + ( lv_value * lv_value * lv_value ).
      ELSE.
        lv_sum = lv_sum + lv_value.
      ENDIF.
    ENDLOOP.

    result = lv_sum.
  ENDMETHOD.

ENDCLASS.