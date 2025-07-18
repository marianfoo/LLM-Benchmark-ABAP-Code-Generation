CLASS z_humaneval_151 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: double_the_difference
      IMPORTING
        lst           TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_151 IMPLEMENTATION.
  METHOD double_the_difference.
    DATA: lv_num       TYPE i,
          lv_remainder TYPE i.

    result = 0.

    LOOP AT lst INTO lv_num.
      IF lv_num <= 0.
        CONTINUE.
      ENDIF.

      lv_remainder = lv_num MOD 2.
      IF lv_remainder = 1.
        result = result + ( lv_num * lv_num ).
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.