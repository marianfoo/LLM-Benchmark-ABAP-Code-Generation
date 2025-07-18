CLASS z_humaneval_121 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: solution
      IMPORTING
        lst           TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.
ENDCLASS.

CLASS z_humaneval_121 IMPLEMENTATION.
  METHOD solution.
    DATA: lv_index TYPE i,
          lv_value TYPE i.

    result = 0.

    LOOP AT lst INTO lv_value.
      lv_index = sy-tabix - 1.
      IF lv_index MOD 2 = 0 AND lv_value MOD 2 = 1.
        result = result + lv_value.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.