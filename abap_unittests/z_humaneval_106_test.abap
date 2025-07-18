CLASS ltcl_test_z_humaneval_106 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_f_5 FOR TESTING,
      test_f_7 FOR TESTING,
      test_f_1 FOR TESTING,
      test_f_3 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_106 IMPLEMENTATION.

  METHOD test_f_5.
    DATA: lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_expected = VALUE #( ( 1 ) ( 2 ) ( 6 ) ( 24 ) ( 15 ) ).
    lv_actual = z_humaneval_106=>f( 5 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected
    ).
  ENDMETHOD.

  METHOD test_f_7.
    DATA: lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_expected = VALUE #( ( 1 ) ( 2 ) ( 6 ) ( 24 ) ( 15 ) ( 720 ) ( 28 ) ).
    lv_actual = z_humaneval_106=>f( 7 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected
    ).
  ENDMETHOD.

  METHOD test_f_1.
    DATA: lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_expected = VALUE #( ( 1 ) ).
    lv_actual = z_humaneval_106=>f( 1 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected
    ).
  ENDMETHOD.

  METHOD test_f_3.
    DATA: lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_expected = VALUE #( ( 1 ) ( 2 ) ( 6 ) ).
    lv_actual = z_humaneval_106=>f( 3 ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected
    ).
  ENDMETHOD.

ENDCLASS.