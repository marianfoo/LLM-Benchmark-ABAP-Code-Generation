CLASS z_humaneval_062 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: derivative
      IMPORTING
        VALUE(xs)     TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_062 IMPLEMENTATION.

  METHOD derivative.
    DATA: lv_index TYPE i,
          lv_coeff TYPE i.

    LOOP AT xs INTO lv_coeff.
      lv_index = sy-tabix - 1.
      IF lv_index > 0.
        result = VALUE #( BASE result ( lv_index * lv_coeff ) ).
      ENDIF.
    ENDLOOP.
  ENDMETHOD.

ENDCLASS.