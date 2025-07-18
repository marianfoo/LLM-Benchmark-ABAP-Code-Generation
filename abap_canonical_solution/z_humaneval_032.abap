CLASS z_humaneval_032 DEFINITION
PUBLIC
FINAL
CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_float_table,
             value TYPE f,
           END OF ty_float_table.
    TYPES: tt_float_table TYPE TABLE OF ty_float_table.

    CLASS-METHODS: find_zero
      IMPORTING
        xs            TYPE tt_float_table
      RETURNING
        VALUE(result) TYPE f.

  PRIVATE SECTION.
    CLASS-METHODS: poly
      IMPORTING
        xs            TYPE tt_float_table
        x             TYPE f
      RETURNING
        VALUE(result) TYPE f.

ENDCLASS.

CLASS z_humaneval_032 IMPLEMENTATION.

  METHOD poly.
    DATA: sum   TYPE f,
          power TYPE f,
          i     TYPE i.

    sum = 0.
    i = 0.

    LOOP AT xs INTO DATA(coeff).
      power = x ** i.
      sum = sum + coeff-value * power.
      i = i + 1.
    ENDLOOP.

    result = sum.
  ENDMETHOD.

  METHOD find_zero.
    DATA: begin_val   TYPE f,
          end_val     TYPE f,
          center      TYPE f,
          poly_begin  TYPE f,
          poly_end    TYPE f,
          poly_center TYPE f.

    begin_val = '-1.0'.
    end_val = '1.0'.

    poly_begin = poly( xs = xs x = begin_val ).
    poly_end = poly( xs = xs x = end_val ).

    WHILE poly_begin * poly_end > 0.
      begin_val = begin_val * '2.0'.
      end_val = end_val * '2.0'.
      poly_begin = poly( xs = xs x = begin_val ).
      poly_end = poly( xs = xs x = end_val ).
    ENDWHILE.

    WHILE end_val - begin_val > + '1E-10'.
      center = ( begin_val + end_val ) / 2.
      poly_center = poly( xs = xs x = center ).
      poly_begin = poly( xs = xs x = begin_val ).

      IF poly_center * poly_begin > 0.
        begin_val = center.
      ELSE.
        end_val = center.
      ENDIF.
    ENDWHILE.

    result = begin_val.
  ENDMETHOD.

ENDCLASS.