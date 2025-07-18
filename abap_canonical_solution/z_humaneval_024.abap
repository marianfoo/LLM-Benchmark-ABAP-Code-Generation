CLASS z_humaneval_024 DEFINITION
PUBLIC
FINAL
CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS largest_divisor
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE i.
ENDCLASS.

CLASS z_humaneval_024 IMPLEMENTATION.
  METHOD largest_divisor.
    DATA: lv_i TYPE i.
    lv_i = n - 1.
    WHILE lv_i > 0.
      IF n MOD lv_i = 0.
        result = lv_i.
        EXIT.
      ENDIF.
      lv_i = lv_i - 1.
    ENDWHILE.
  ENDMETHOD.
ENDCLASS.