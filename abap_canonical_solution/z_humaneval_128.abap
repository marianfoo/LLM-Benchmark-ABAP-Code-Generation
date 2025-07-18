CLASS z_humaneval_128 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: prod_signs
      IMPORTING arr           TYPE int4_table
      RETURNING VALUE(result) TYPE int4.

ENDCLASS.

CLASS z_humaneval_128 IMPLEMENTATION.

  METHOD prod_signs.
    DATA: lv_prod      TYPE int4,
          lv_sum       TYPE int4,
          lv_neg_count TYPE int4.

    IF lines( arr ) = 0.
      result = 0.
      RETURN.
    ENDIF.

    LOOP AT arr INTO DATA(lv_value).
      IF lv_value = 0.
        result = 0.
        RETURN.
      ENDIF.

      lv_sum = lv_sum + abs( lv_value ).

      IF lv_value < 0.
        lv_neg_count = lv_neg_count + 1.
      ENDIF.
    ENDLOOP.

    lv_prod = COND #( WHEN lv_neg_count MOD 2 = 0 THEN 1 ELSE -1 ).
    result = lv_prod * lv_sum.

  ENDMETHOD.

ENDCLASS.