CLASS z_humaneval_077 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: iscube
      IMPORTING
        VALUE(a)      TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_077 IMPLEMENTATION.

  METHOD iscube.
    DATA: lv_abs_a        TYPE i,
          lv_cube_root    TYPE f,
          lv_rounded_root TYPE i,
          lv_cube         TYPE i.

    lv_abs_a = abs( a ).
    lv_cube_root = lv_abs_a ** ( 1 / 3 ).
    lv_rounded_root = nmax( val1 = trunc( lv_cube_root + '0.5' ) val2 = 0 ).
    lv_cube = lv_rounded_root ** 3.

    IF lv_cube = lv_abs_a.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.