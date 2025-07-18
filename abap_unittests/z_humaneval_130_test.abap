CLASS test_z_humaneval_130 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_tri_3 FOR TESTING,
      test_tri_4 FOR TESTING,
      test_tri_5 FOR TESTING,
      test_tri_6 FOR TESTING,
      test_tri_7 FOR TESTING,
      test_tri_8 FOR TESTING,
      test_tri_9 FOR TESTING,
      test_tri_20 FOR TESTING,
      test_tri_0 FOR TESTING,
      test_tri_1 FOR TESTING.

ENDCLASS.

CLASS test_z_humaneval_130 IMPLEMENTATION.

  METHOD test_tri_3.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ).
    actual = z_humaneval_130=>tri( 3 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_4.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ( 3 ) ).
    actual = z_humaneval_130=>tri( 4 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_5.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ( 3 ) ( 15 ) ).
    actual = z_humaneval_130=>tri( 5 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_6.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ( 3 ) ( 15 ) ( 4 ) ).
    actual = z_humaneval_130=>tri( 6 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_7.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ( 3 ) ( 15 ) ( 4 ) ( 24 ) ).
    actual = z_humaneval_130=>tri( 7 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_8.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ( 3 ) ( 15 ) ( 4 ) ( 24 ) ( 5 ) ).
    actual = z_humaneval_130=>tri( 8 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_9.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ( 3 ) ( 15 ) ( 4 ) ( 24 ) ( 5 ) ( 35 ) ).
    actual = z_humaneval_130=>tri( 9 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_20.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 8 ) ( 3 ) ( 15 ) ( 4 ) ( 24 ) ( 5 ) ( 35 ) ( 6 ) ( 48 ) ( 7 ) ( 63 ) ( 8 ) ( 80 ) ( 9 ) ( 99 ) ( 10 ) ( 120 ) ( 11 ) ).
    actual = z_humaneval_130=>tri( 20 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_0.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ).
    actual = z_humaneval_130=>tri( 0 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_tri_1.
    DATA: expected TYPE int4_table,
          actual   TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ).
    actual = z_humaneval_130=>tri( 1 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

ENDCLASS.